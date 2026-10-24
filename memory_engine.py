"""Local SQLite memory with bounded hybrid keyword and vector search."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import re
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Generator

import httpx
import numpy as np


SEARCH_CANDIDATE_LIMIT = 100
SEARCH_KEYWORD_CANDIDATE_LIMIT = SEARCH_CANDIDATE_LIMIT // 2
RETENTION_REPORT_MAX_LIMIT = 100
PENDING_MEMORY_BATCH_SIZE = 2
PENDING_MEMORY_CHUNK_CHARS = 5_000
_SEARCH_TOKEN_PATTERN = re.compile(r"[^\W_]+(?:[_+#.-][^\W_]+)*|[0-9]+", re.UNICODE)
_SEARCH_STOP_WORDS = frozenset(
    {"a", "an", "and", "are", "can", "do", "does", "for", "how", "i", "in",
     "is", "it", "me", "of", "on", "or", "please", "the", "this", "to",
     "was", "what", "when", "where", "why", "with", "you"}
)
_CRAWLER_NOISE_PATTERN = re.compile(
    r"\b(?:accept all cookies|cookie preferences|privacy settings|"
    r"subscribe to our newsletter|sign up for our newsletter|"
    r"share this page|was this page helpful|send feedback|"
    r"advertisement|sponsored content|related articles)\b",
    re.IGNORECASE,
)
_TECHNICAL_CONTENT_PATTERN = re.compile(
    r"\b(?:api|async|class|code|command|component|database|endpoint|"
    r"exception|function|http|import|install|javascript|json|method|"
    r"module|parameter|package|python|query|request|response|return|"
    r"server|sql|syntax|typescript|type|variable)\b|```|"
    r"^\s{0,3}(?:def|class|import|from|const|let|var|function)\b",
    re.IGNORECASE | re.MULTILINE,
)
_PENDING_BOILERPLATE_PATTERN = re.compile(
    r"\b(?:accept all cookies|cookie preferences|privacy settings|"
    r"subscribe to our newsletter|sign up for our newsletter|"
    r"share this page|was this page helpful|send feedback|"
    r"advertisement|sponsored content|related articles)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class MemoryRecord:
    """A stored text item and its metadata."""

    id: int
    text: str
    source: str
    created_at: str
    similarity: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable memory data."""
        return {
            "id": self.id,
            "text": self.text,
            "source": self.source,
            "created_at": self.created_at,
            "similarity": self.similarity,
        }


class MemoryEngine:
    """Persist local notes and perform semantic nearest-neighbor searches."""

    def __init__(
        self,
        database_path: Path,
        ollama_base_url: str,
        embedding_model: str,
        logger: logging.Logger | None = None,
        crawl_queue_max_pending: int = 100,
        ollama_keep_alive: str = "1m",
        ollama_num_thread: int = 4,
        ollama_request_semaphore: asyncio.Semaphore | None = None,
    ) -> None:
        self.database_path = database_path
        self.ollama_base_url = ollama_base_url.rstrip("/")
        self.embedding_model = embedding_model
        self.ollama_keep_alive = ollama_keep_alive
        if not 1 <= ollama_num_thread <= 8:
            raise ValueError("Ollama thread count must be between 1 and 8.")
        self.ollama_num_thread = ollama_num_thread
        self.logger = logger or logging.getLogger("omnimind.memory")
        if not 1 <= crawl_queue_max_pending <= 500:
            raise ValueError("Crawl queue capacity must be between 1 and 500.")
        self.crawl_queue_max_pending = crawl_queue_max_pending
        self._lock = asyncio.Lock()
        self._embedding_semaphore = (
            ollama_request_semaphore or asyncio.Semaphore(1)
        )
        self._connection_slots = threading.BoundedSemaphore(2)

    async def initialize(self) -> None:
        """Create the database schema and enable durable local journaling."""
        await asyncio.to_thread(self._initialize_sync)

    def _connect(self) -> sqlite3.Connection:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.database_path, timeout=10.0)
        try:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute("PRAGMA cache_size=-2048")
            connection.execute("PRAGMA temp_store=FILE")
            connection.execute("PRAGMA mmap_size=0")
            connection.execute("PRAGMA journal_size_limit=16777216")
            connection.execute("PRAGMA wal_autocheckpoint=500")
        except sqlite3.Error:
            connection.close()
            raise
        return connection

    @contextmanager
    def _connection(self) -> Generator[sqlite3.Connection, None, None]:
        with self._connection_slots:
            connection = self._connect()
            try:
                with connection:
                    yield connection
            finally:
                connection.close()

    def _initialize_sync(self) -> None:
        with self._connection() as connection:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    text TEXT NOT NULL,
                    source TEXT NOT NULL DEFAULT 'user',
                    embedding TEXT,
                    created_at TEXT NOT NULL,
                    user_authored INTEGER NOT NULL DEFAULT 1
                )"""
            )
            memory_columns = {
                str(row["name"])
                for row in connection.execute("PRAGMA table_info(memories)")
            }
            if "user_authored" not in memory_columns:
                connection.execute(
                    "ALTER TABLE memories ADD COLUMN user_authored "
                    "INTEGER NOT NULL DEFAULT 1"
                )
                connection.execute(
                    "UPDATE memories SET user_authored = 0 "
                    "WHERE source LIKE 'crawl:%'"
                )
            for name, declaration in (
                ("document_url", "TEXT"),
                ("source_version", "TEXT"),
                ("expires_at", "TEXT"),
                ("processing_status", "TEXT NOT NULL DEFAULT 'pending'"),
                ("category", "TEXT NOT NULL DEFAULT 'general'"),
            ):
                if name not in memory_columns:
                    connection.execute(
                        f"ALTER TABLE memories ADD COLUMN {name} {declaration}"
                    )
            connection.execute(
                "UPDATE memories SET processing_status = 'done' "
                "WHERE embedding IS NOT NULL AND processing_status = 'pending'"
            )
            connection.execute(
                "UPDATE memories SET category = CASE "
                "WHEN source LIKE 'techdoc:%' THEN 'technical_documentation' "
                "WHEN source LIKE 'crawl:%' THEN 'crawled_content' "
                "WHEN source = 'user' THEN 'user_note' "
                "ELSE 'other' END WHERE category = 'general'"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_processing_status "
                "ON memories(processing_status, id)"
            )
            connection.execute(
                "DROP INDEX IF EXISTS idx_memories_created_at"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_created_at_id "
                "ON memories(created_at DESC, id DESC)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_retention "
                "ON memories(user_authored, created_at, id)"
            )
            connection.execute(
                """CREATE TABLE IF NOT EXISTS technical_documents (
                    url TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    host TEXT NOT NULL,
                    indexed_at TEXT NOT NULL,
                    chunk_count INTEGER NOT NULL CHECK (chunk_count > 0),
                    source_version TEXT NOT NULL DEFAULT 'legacy',
                    expires_at TEXT
                )"""
            )
            document_columns = {
                str(row["name"])
                for row in connection.execute("PRAGMA table_info(technical_documents)")
            }
            if "source_version" not in document_columns:
                connection.execute(
                    "ALTER TABLE technical_documents ADD COLUMN source_version "
                    "TEXT NOT NULL DEFAULT 'legacy'"
                )
            if "expires_at" not in document_columns:
                connection.execute(
                    "ALTER TABLE technical_documents ADD COLUMN expires_at TEXT"
                )
            connection.execute(
                "UPDATE technical_documents SET expires_at = "
                "strftime('%Y-%m-%dT%H:%M:%f+00:00', indexed_at, '+90 days') "
                "WHERE expires_at IS NULL"
            )
            connection.execute(
                "UPDATE memories SET document_url = ("
                "SELECT MIN(url) FROM technical_documents "
                "WHERE 'techdoc:' || host = memories.source "
                "HAVING COUNT(*) = 1"
                ") WHERE source LIKE 'techdoc:%' AND document_url IS NULL "
                "AND (SELECT COUNT(*) FROM technical_documents "
                "WHERE 'techdoc:' || host = memories.source) = 1"
            )
            connection.execute(
                "UPDATE memories SET source_version = 'legacy' "
                "WHERE source LIKE 'techdoc:%' AND source_version IS NULL"
            )
            connection.execute(
                "UPDATE memories SET expires_at = ("
                "SELECT expires_at FROM technical_documents "
                "WHERE technical_documents.url = memories.document_url"
                ") WHERE source LIKE 'techdoc:%' AND expires_at IS NULL "
                "AND document_url IS NOT NULL"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_technical_documents_indexed "
                "ON technical_documents(indexed_at DESC)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_document_url "
                "ON memories(document_url)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_expires_at "
                "ON memories(expires_at)"
            )
            connection.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(text)"
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS memory_search_index_state "
                "(id INTEGER PRIMARY KEY CHECK (id = 1), initialized INTEGER NOT NULL)"
            )
            search_index_initialized = connection.execute(
                "SELECT 1 FROM memory_search_index_state WHERE id = 1"
            ).fetchone()
            if search_index_initialized is None:
                connection.execute(
                    "INSERT INTO memories_fts(rowid, text) "
                    "SELECT m.id, m.text FROM memories m WHERE NOT EXISTS "
                    "(SELECT 1 FROM memories_fts f WHERE f.rowid = m.id)"
                )
                connection.execute(
                    "INSERT INTO memory_search_index_state(id, initialized) VALUES (1, 1)"
                )
            connection.executescript(
                """
                CREATE TRIGGER IF NOT EXISTS memories_fts_insert
                AFTER INSERT ON memories BEGIN
                    INSERT INTO memories_fts(rowid, text) VALUES (new.id, new.text);
                END;
                CREATE TRIGGER IF NOT EXISTS memories_fts_delete
                AFTER DELETE ON memories BEGIN
                    DELETE FROM memories_fts WHERE rowid = old.id;
                END;
                CREATE TRIGGER IF NOT EXISTS memories_fts_update
                AFTER UPDATE OF text ON memories BEGIN
                    DELETE FROM memories_fts WHERE rowid = old.id;
                    INSERT INTO memories_fts(rowid, text) VALUES (new.id, new.text);
                END;
                """
            )
            connection.execute(
                """CREATE TABLE IF NOT EXISTS crawl_queue (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    url TEXT NOT NULL UNIQUE,
                    status TEXT NOT NULL CHECK (
                        status IN ('queued', 'processing', 'visited', 'failed')
                    ),
                    discovered_from TEXT,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )"""
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_crawl_queue_status "
                "ON crawl_queue(status, id)"
            )
            connection.execute(
                "UPDATE crawl_queue SET status = 'queued', updated_at = ? "
                "WHERE status = 'processing'",
                (datetime.now(timezone.utc).isoformat(),),
            )

    async def embed(self, text: str) -> list[float] | None:
        """Ask the local Ollama server for an embedding; return None if offline."""
        normalized = text.strip()
        if not normalized:
            return None
        timeout = httpx.Timeout(45.0, connect=3.0)
        async with self._embedding_semaphore:
            async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
                try:
                    response = await client.post(
                        f"{self.ollama_base_url}/api/embed",
                        json={
                            "model": self.embedding_model,
                            "input": normalized,
                            "keep_alive": self.ollama_keep_alive,
                            "options": {"num_thread": self.ollama_num_thread},
                        },
                    )
                    if response.status_code == 404:
                        response = await client.post(
                            f"{self.ollama_base_url}/api/embeddings",
                            json={
                                "model": self.embedding_model,
                                "prompt": normalized,
                                "keep_alive": self.ollama_keep_alive,
                                "options": {"num_thread": self.ollama_num_thread},
                            },
                        )
                    response.raise_for_status()
                    payload = response.json()
                    raw = payload.get("embeddings")
                    if raw:
                        vector = raw[0]
                    else:
                        vector = payload.get("embedding")
                    if not isinstance(vector, list) or not vector:
                        raise ValueError("Ollama returned no embedding vector.")
                    result = [float(value) for value in vector]
                    if not all(math.isfinite(value) for value in result):
                        raise ValueError("Ollama returned a non-finite embedding value.")
                    return result
                except (httpx.HTTPError, ValueError, TypeError, KeyError) as error:
                    self.logger.warning("Local embedding unavailable: %s", error)
                    return None

    async def add_memory(
        self,
        text: str,
        source: str = "user",
        *,
        user_authored: bool = True,
        require_embedding: bool = False,
    ) -> MemoryRecord:
        """Store a memory; new records are protected from pruning by default."""
        normalized = text.strip()
        if not normalized:
            raise ValueError("Memory text cannot be empty.")
        if len(normalized) > 20_000:
            raise ValueError("Memory text cannot exceed 20,000 characters.")
        if not isinstance(user_authored, bool):
            raise ValueError("user_authored must be a boolean.")
        clean_source = source.strip()[:80] or "user"
        embedding = await self.embed(normalized)
        if require_embedding and embedding is None:
            raise RuntimeError("Memory was not stored because its embedding is unavailable.")
        created_at = datetime.now(timezone.utc).isoformat()
        async with self._lock:
            identifier = await asyncio.to_thread(
                self._insert_sync,
                normalized,
                clean_source,
                embedding,
                created_at,
                user_authored,
            )
        return MemoryRecord(identifier, normalized, clean_source, created_at)

    async def add_memory_batch(
        self,
        texts: list[str],
        source: str,
        *,
        user_authored: bool = False,
        crawl_url: str | None = None,
    ) -> int:
        """Embed and atomically store a small group of related memory chunks."""
        if not texts or len(texts) > 4:
            raise ValueError("A memory batch must contain 1 to 4 chunks.")
        normalized = [text.strip() for text in texts]
        if any(not text or len(text) > 5000 for text in normalized):
            raise ValueError("Memory chunks must contain 1 to 5000 characters.")
        if not isinstance(user_authored, bool):
            raise ValueError("user_authored must be a boolean.")
        clean_source = source.strip()[:80] or "user"
        embeddings: list[list[float]] = []
        for text in normalized:
            embedding = await self.embed(text)
            if embedding is None:
                raise RuntimeError(
                    "Memory batch was not stored because embeddings are unavailable."
                )
            embeddings.append(embedding)

        created_at = datetime.now(timezone.utc).isoformat()
        async with self._lock:
            return await asyncio.to_thread(
                self._insert_memory_batch_sync,
                normalized,
                clean_source,
                embeddings,
                created_at,
                user_authored,
                crawl_url,
            )

    def _insert_memory_batch_sync(
        self,
        texts: list[str],
        source: str,
        embeddings: list[list[float]],
        created_at: str,
        user_authored: bool,
        crawl_url: str | None,
    ) -> int:
        category = _memory_category(source)
        with self._connection() as connection:
            connection.executemany(
                "INSERT INTO memories"
                "(text, source, embedding, created_at, user_authored, "
                "processing_status, category) "
                "VALUES (?, ?, ?, ?, ?, 'done', ?)",
                [
                    (
                        text,
                        source,
                        json.dumps(embedding),
                        created_at,
                        int(user_authored),
                        category,
                    )
                    for text, embedding in zip(texts, embeddings)
                ],
            )
            if crawl_url is not None:
                cursor = connection.execute(
                    "UPDATE crawl_queue SET status = 'visited', error = NULL, "
                    "updated_at = ? WHERE url = ? AND status = 'processing'",
                    (created_at, crawl_url),
                )
                if cursor.rowcount != 1:
                    raise RuntimeError(
                        "Could not finalize the active crawl-queue entry."
                    )
        return len(texts)

    async def technical_document_indexed(self, url: str) -> bool:
        """Check whether a curated technical documentation URL is already indexed."""
        return await asyncio.to_thread(self._technical_document_indexed_sync, url)

    def _technical_document_indexed_sync(self, url: str) -> bool:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT 1 FROM technical_documents WHERE url = ?",
                (url,),
            ).fetchone()
        return row is not None

    async def technical_document_status(
        self, url: str, ttl_days: int
    ) -> dict[str, Any] | None:
        """Return bounded freshness metadata for a previously indexed URL."""
        return await asyncio.to_thread(
            self._technical_document_status_sync, url, ttl_days
        )

    def _technical_document_status_sync(
        self, url: str, ttl_days: int
    ) -> dict[str, Any] | None:
        now = datetime.now(timezone.utc).isoformat()
        with self._connection() as connection:
            row = connection.execute(
                "SELECT url, title, host, indexed_at, chunk_count, "
                "source_version, expires_at FROM technical_documents WHERE url = ?",
                (url,),
            ).fetchone()
        if row is None:
            return None
        expires_at = row["expires_at"]
        if not expires_at:
            indexed_at = datetime.fromisoformat(str(row["indexed_at"]))
            expires_at = (indexed_at + timedelta(days=ttl_days)).isoformat()
        return {
            "url": str(row["url"]),
            "title": str(row["title"]),
            "host": str(row["host"]),
            "indexed_at": str(row["indexed_at"]),
            "chunk_count": int(row["chunk_count"]),
            "source_version": str(row["source_version"]),
            "expires_at": str(expires_at),
            "stale": str(expires_at) <= now,
        }

    async def technical_freshness_report(
        self, ttl_days: int, limit: int = 50
    ) -> dict[str, Any]:
        """Report stale indexed documents without deleting or loading all records."""
        bounded_limit = max(1, min(limit, RETENTION_REPORT_MAX_LIMIT))
        now = datetime.now(timezone.utc).isoformat()
        return await asyncio.to_thread(
            self._technical_freshness_report_sync, ttl_days, bounded_limit, now
        )

    def _technical_freshness_report_sync(
        self, ttl_days: int, limit: int, now: str
    ) -> dict[str, Any]:
        cutoff = (datetime.fromisoformat(now) - timedelta(days=ttl_days)).isoformat()
        with self._connection() as connection:
            stale_count = int(
                connection.execute(
                    "SELECT COUNT(*) FROM technical_documents "
                    "WHERE COALESCE(expires_at, "
                    "strftime('%Y-%m-%dT%H:%M:%f+00:00', indexed_at, '+90 days')) <= ?",
                    (now,),
                ).fetchone()[0]
            )
            rows = connection.execute(
                "SELECT url, title, host, indexed_at, source_version, expires_at "
                "FROM technical_documents WHERE "
                "COALESCE(expires_at, ?) <= ? "
                "ORDER BY indexed_at ASC LIMIT ?",
                (cutoff, now, limit),
            ).fetchall()
        return {
            "ttl_days": ttl_days,
            "stale_count": stale_count,
            "documents": [
                {
                    "url": str(row["url"]),
                    "title": str(row["title"]),
                    "host": str(row["host"]),
                    "indexed_at": str(row["indexed_at"]),
                    "source_version": str(row["source_version"]),
                    "expires_at": str(row["expires_at"] or cutoff),
                }
                for row in rows
            ],
        }

    async def refresh_technical_document(
        self, url: str, source_version: str, ttl_days: int
    ) -> bool:
        """Extend freshness without re-embedding when fetched content is unchanged."""
        now = datetime.now(timezone.utc)
        expires_at = (now + timedelta(days=ttl_days)).isoformat()
        async with self._lock:
            return await asyncio.to_thread(
                self._refresh_technical_document_sync,
                url,
                source_version,
                now.isoformat(),
                expires_at,
            )

    def _refresh_technical_document_sync(
        self, url: str, source_version: str, indexed_at: str, expires_at: str
    ) -> bool:
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT source_version FROM technical_documents WHERE url = ?",
                (url,),
            ).fetchone()
            if row is None or str(row["source_version"]) != source_version:
                return False
            connection.execute(
                "UPDATE technical_documents SET indexed_at = ?, expires_at = ? "
                "WHERE url = ?",
                (indexed_at, expires_at, url),
            )
            connection.execute(
                "UPDATE memories SET created_at = ?, expires_at = ?, "
                "source_version = ? WHERE document_url = ?",
                (indexed_at, expires_at, source_version, url),
            )
            return True

    async def index_technical_document(
        self,
        *,
        url: str,
        title: str,
        host: str,
        chunks: list[str],
        max_records: int,
        source_version: str = "legacy",
        ttl_days: int = 90,
        refresh: bool = False,
    ) -> int:
        """Embed and atomically index one bounded technical document.

        Embeddings are prepared sequentially before the write transaction. The
        document URL is unique, and the technical-memory cap is enforced inside
        the transaction so concurrent requests cannot exceed it.
        """
        if not url or not title or not host:
            raise ValueError("Technical document URL, title, and host are required.")
        if not 1 <= max_records <= 2000:
            raise ValueError("Technical memory capacity must be between 1 and 2000.")
        if not chunks or len(chunks) > 10:
            raise ValueError("A technical document must contain 1 to 10 chunks.")
        normalized_chunks = [chunk.strip() for chunk in chunks]
        if any(not chunk or len(chunk) > 5000 for chunk in normalized_chunks):
            raise ValueError("Technical document chunks must contain 1 to 5000 characters.")

        embeddings: list[list[float]] = []
        for chunk in normalized_chunks:
            vector = await self.embed(chunk)
            if vector is None:
                raise RuntimeError(
                    "Technical document was not indexed because Ollama embeddings "
                    "are unavailable."
                )
            embeddings.append(vector)

        created = datetime.now(timezone.utc)
        created_at = created.isoformat()
        expires_at = (created + timedelta(days=ttl_days)).isoformat()
        source = f"techdoc:{host}"[:80]
        async with self._lock:
            return await asyncio.to_thread(
                self._index_technical_document_sync,
                url,
                title[:300],
                host[:253],
                source,
                normalized_chunks,
                embeddings,
                max_records,
                created_at,
                expires_at,
                source_version[:300],
                refresh,
            )

    async def technical_document_count(self) -> int:
        """Return the number of indexed curated technical documentation pages."""
        return await asyncio.to_thread(self._technical_document_count_sync)

    def _technical_document_count_sync(self) -> int:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT COUNT(*) FROM technical_documents"
            ).fetchone()
        return int(row[0])

    async def technical_memory_count(self) -> int:
        """Return stored technical chunks for enforcing/reporting the configured cap."""
        return await self.count("techdoc:")

    async def pending_memory_status(self) -> dict[str, Any]:
        """Return bounded aggregate counts for locally stored learning records."""
        return await asyncio.to_thread(self._pending_memory_status_sync)

    def _pending_memory_status_sync(self) -> dict[str, Any]:
        with self._connection() as connection:
            memory_row = connection.execute(
                "SELECT COUNT(*) AS total, "
                "SUM(CASE WHEN processing_status = 'done' THEN 1 ELSE 0 END) AS learned, "
                "SUM(CASE WHEN processing_status = 'pending' THEN 1 ELSE 0 END) AS pending, "
                "SUM(CASE WHEN processing_status = 'ignored' THEN 1 ELSE 0 END) AS ignored "
                "FROM memories"
            ).fetchone()
            category_rows = connection.execute(
                "SELECT category, COUNT(*) AS count FROM memories "
                "GROUP BY category ORDER BY category LIMIT 20"
            ).fetchall()
            queue_row = connection.execute(
                "SELECT SUM(CASE WHEN status IN ('queued', 'processing') "
                "THEN 1 ELSE 0 END) AS pending_urls FROM crawl_queue"
            ).fetchone()
        return {
            "total": int(memory_row["total"] or 0),
            "learned": int(memory_row["learned"] or 0),
            "pending": int(memory_row["pending"] or 0),
            "ignored": int(memory_row["ignored"] or 0),
            "pending_urls": int(queue_row["pending_urls"] or 0),
            "categories": {
                str(row["category"]): int(row["count"])
                for row in category_rows
            },
        }

    async def pending_memory_batch(
        self, limit: int = PENDING_MEMORY_BATCH_SIZE
    ) -> list[dict[str, Any]]:
        """Read a small oldest-first batch of pending records without embeddings."""
        bounded_limit = max(1, min(limit, PENDING_MEMORY_BATCH_SIZE))
        return await asyncio.to_thread(
            self._pending_memory_batch_sync, bounded_limit
        )

    def _pending_memory_batch_sync(self, limit: int) -> list[dict[str, Any]]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT id, text, source, created_at, user_authored, "
                "document_url, source_version, expires_at "
                "FROM memories WHERE processing_status = 'pending' "
                "ORDER BY id LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    async def finish_pending_memory(
        self,
        record: dict[str, Any],
        chunks: list[str],
        embeddings: list[list[float]],
    ) -> bool:
        """Atomically update one stored record and add its bounded text chunks."""
        if not chunks or len(chunks) > 4 or len(chunks) != len(embeddings):
            raise ValueError("Pending memory chunks and embeddings are inconsistent.")
        if any(not chunk or len(chunk) > PENDING_MEMORY_CHUNK_CHARS for chunk in chunks):
            raise ValueError("Pending memory chunks exceed the configured bound.")
        return await asyncio.to_thread(
            self._finish_pending_memory_sync, record, chunks, embeddings
        )

    def _finish_pending_memory_sync(
        self,
        record: dict[str, Any],
        chunks: list[str],
        embeddings: list[list[float]],
    ) -> bool:
        category = _memory_category(str(record["source"]))
        created_at = datetime.now(timezone.utc).isoformat()
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                "UPDATE memories SET text = ?, embedding = ?, "
                "processing_status = 'done', category = ? "
                "WHERE id = ? AND processing_status = 'pending'",
                (
                    chunks[0],
                    json.dumps(embeddings[0]),
                    category,
                    int(record["id"]),
                ),
            )
            if cursor.rowcount != 1:
                return False
            extra_chunks = chunks[1:]
            if extra_chunks:
                connection.executemany(
                    "INSERT INTO memories"
                    "(text, source, embedding, created_at, user_authored, "
                    "document_url, source_version, expires_at, processing_status, category) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'done', ?)",
                    [
                        (
                            chunk,
                            str(record["source"]),
                            json.dumps(embedding),
                            created_at,
                            int(record["user_authored"]),
                            record["document_url"],
                            record["source_version"],
                            record["expires_at"],
                            category,
                        )
                        for chunk, embedding in zip(extra_chunks, embeddings[1:])
                    ],
                )
                if record["document_url"]:
                    connection.execute(
                        "UPDATE technical_documents SET chunk_count = chunk_count + ? "
                        "WHERE url = ?",
                        (len(extra_chunks), record["document_url"]),
                    )
        return True

    async def ignore_pending_memory(self, identifier: int) -> bool:
        """Mark an empty/boilerplate-only pending record without embedding it."""
        return await asyncio.to_thread(self._ignore_pending_memory_sync, identifier)

    def _ignore_pending_memory_sync(self, identifier: int) -> bool:
        with self._connection() as connection:
            cursor = connection.execute(
                "UPDATE memories SET processing_status = 'ignored' "
                "WHERE id = ? AND processing_status = 'pending'",
                (identifier,),
            )
        return cursor.rowcount == 1

    def _index_technical_document_sync(
        self,
        url: str,
        title: str,
        host: str,
        source: str,
        chunks: list[str],
        embeddings: list[list[float]],
        max_records: int,
        indexed_at: str,
        expires_at: str,
        source_version: str,
        refresh: bool,
    ) -> int:
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT 1 FROM technical_documents WHERE url = ?", (url,)
            ).fetchone()
            if existing and not refresh:
                return 0
            if existing:
                connection.execute(
                    "DELETE FROM memories WHERE document_url = ?", (url,)
                )

            row = connection.execute(
                "SELECT COUNT(*) FROM memories WHERE source LIKE 'techdoc:%'"
            ).fetchone()
            remaining_capacity = max(0, max_records - int(row[0]))
            chunk_count = min(remaining_capacity, len(chunks))
            if chunk_count == 0:
                return 0

            connection.executemany(
                "INSERT INTO memories"
                "(text, source, embedding, created_at, user_authored, "
                "document_url, source_version, expires_at, processing_status, category) "
                "VALUES (?, ?, ?, ?, 0, ?, ?, ?, 'done', 'technical_documentation')",
                [
                    (
                        chunks[index],
                        source,
                        json.dumps(embeddings[index]),
                        indexed_at,
                        url,
                        source_version,
                        expires_at,
                    )
                    for index in range(chunk_count)
                ],
            )
            if existing:
                connection.execute(
                    "UPDATE technical_documents SET title = ?, host = ?, "
                    "indexed_at = ?, chunk_count = ?, source_version = ?, expires_at = ? "
                    "WHERE url = ?",
                    (
                        title,
                        host,
                        indexed_at,
                        chunk_count,
                        source_version,
                        expires_at,
                        url,
                    ),
                )
            else:
                connection.execute(
                    "INSERT INTO technical_documents"
                    "(url, title, host, indexed_at, chunk_count, source_version, expires_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        url,
                        title,
                        host,
                        indexed_at,
                        chunk_count,
                        source_version,
                        expires_at,
                    ),
                )
            return chunk_count

    def _insert_sync(
        self,
        text: str,
        source: str,
        embedding: list[float] | None,
        created_at: str,
        user_authored: bool,
    ) -> int:
        with self._connection() as connection:
            cursor = connection.execute(
                "INSERT INTO memories"
                "(text, source, embedding, created_at, user_authored, "
                "processing_status, category) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    text,
                    source,
                    json.dumps(embedding) if embedding is not None else None,
                    created_at,
                    int(user_authored),
                    "done" if embedding is not None else "pending",
                    _memory_category(source),
                ),
            )
            return int(cursor.lastrowid)

    async def search(self, query: str, limit: int = 8) -> list[MemoryRecord]:
        """Rank a bounded set of recent candidates, falling back to recent memories.

        Candidate retrieval is capped in SQLite before embedding JSON is loaded,
        so Python memory use does not grow with the full database size.
        """
        normalized = query.strip()
        if not normalized:
            return []
        bounded_limit = max(1, min(limit, 50))
        query_tokens = _search_tokens(normalized)
        fts_query = " OR ".join(
            f'"{token.replace(chr(34), chr(34) * 2)}"' for token in query_tokens
        )
        query_vector, rows = await asyncio.gather(
            self.embed(normalized),
            asyncio.to_thread(
                self._read_search_candidates_sync,
                SEARCH_CANDIDATE_LIMIT,
                fts_query,
            ),
        )
        return await asyncio.to_thread(
            self._rank_search_candidates,
            rows,
            query_vector,
            bounded_limit,
            query_tokens,
        )

    @staticmethod
    def _rank_search_candidates(
        rows: list[sqlite3.Row],
        query_vector: list[float] | None,
        limit: int,
        query_tokens: list[str] | None = None,
    ) -> list[MemoryRecord]:
        """Blend semantic and lexical scores for the bounded candidate set."""
        query_tokens = query_tokens or []
        ranked: list[tuple[float, sqlite3.Row]] = []
        for row in rows:
            if _is_low_quality_memory(str(row["text"]), str(row["source"])):
                continue
            lexical_score = _lexical_score(str(row["text"]), query_tokens)
            semantic_score = 0.0
            raw_vector = row["embedding"]
            if query_vector is not None and raw_vector:
                try:
                    vector = json.loads(raw_vector)
                    similarity = _cosine_similarity(query_vector, vector)
                except (ValueError, TypeError, json.JSONDecodeError):
                    similarity = None
                if similarity is not None:
                    semantic_score = max(0.0, similarity)
            if query_vector is not None:
                score = 0.55 * semantic_score + 0.45 * lexical_score
            else:
                score = lexical_score
            if score > 0:
                ranked.append((score, row))
        if ranked:
            ranked.sort(key=lambda item: (item[0], item[1]["created_at"]), reverse=True)
            return [
                _record(row, round(score, 5))
                for score, row in ranked[:limit]
            ]
        usable_rows = [
            row
            for row in rows
            if not _is_low_quality_memory(str(row["text"]), str(row["source"]))
        ]
        return [_record(row, None) for row in usable_rows[:limit]]

    async def history(self, limit: int = 50) -> list[MemoryRecord]:
        """Return newest memories without embedding payloads."""
        bounded_limit = max(1, min(limit, 200))
        rows = await asyncio.to_thread(self._history_sync, bounded_limit)
        return [_record(row, None) for row in rows]

    async def count(self, source_prefix: str | None = None) -> int:
        """Return total memory count, optionally filtered by a source prefix."""
        return await asyncio.to_thread(self._count_sync, source_prefix)

    async def enqueue_crawl_urls(
        self, urls: list[str], discovered_from: str | None = None
    ) -> int:
        """Persist new crawl URLs while ignoring URLs already queued or handled."""
        if not urls:
            return 0
        now = datetime.now(timezone.utc).isoformat()
        async with self._lock:
            return await asyncio.to_thread(
                self._enqueue_crawl_urls_sync, urls, discovered_from, now
            )

    def _enqueue_crawl_urls_sync(
        self, urls: list[str], discovered_from: str | None, created_at: str
    ) -> int:
        with self._connection() as connection:
            before = connection.total_changes
            connection.executemany(
                "INSERT OR IGNORE INTO crawl_queue"
                "(url, status, discovered_from, created_at, updated_at) "
                "SELECT ?, 'queued', ?, ?, ? WHERE "
                "(SELECT COUNT(*) FROM crawl_queue "
                "WHERE status = 'queued') < ?",
                [
                    (
                        url,
                        discovered_from,
                        created_at,
                        created_at,
                        self.crawl_queue_max_pending,
                    )
                    for url in urls
                ],
            )
            return connection.total_changes - before

    async def claim_next_crawl_url(self) -> str | None:
        """Atomically claim the oldest queued crawl URL."""
        async with self._lock:
            return await asyncio.to_thread(self._claim_next_crawl_url_sync)

    def _claim_next_crawl_url_sync(self) -> str | None:
        now = datetime.now(timezone.utc).isoformat()
        with self._connection() as connection:
            row = connection.execute(
                "SELECT id, url FROM crawl_queue WHERE status = 'queued' "
                "ORDER BY id LIMIT 1"
            ).fetchone()
            if row is None:
                return None
            cursor = connection.execute(
                "UPDATE crawl_queue SET status = 'processing', attempts = attempts + 1, "
                "updated_at = ? WHERE id = ? AND status = 'queued'",
                (now, int(row["id"])),
            )
            return str(row["url"]) if cursor.rowcount == 1 else None

    async def finish_crawl_url(self, url: str) -> None:
        """Mark a successfully processed crawl URL as permanently visited."""
        await self._update_crawl_url(url, "visited")

    async def release_crawl_url(self, url: str) -> None:
        """Return an interrupted in-progress URL to the pending queue."""
        await self._update_crawl_url(url, "queued")

    async def fail_crawl_url(self, url: str, error: str) -> None:
        """Record a failed crawl without retrying it automatically."""
        await self._update_crawl_url(url, "failed", error[:1000])

    async def _update_crawl_url(
        self, url: str, status: str, error: str | None = None
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        async with self._lock:
            await asyncio.to_thread(self._update_crawl_url_sync, url, status, error, now)

    def _update_crawl_url_sync(
        self, url: str, status: str, error: str | None, updated_at: str
    ) -> None:
        with self._connection() as connection:
            connection.execute(
                "UPDATE crawl_queue SET status = ?, error = ?, updated_at = ? "
                "WHERE url = ?",
                (status, error, updated_at, url),
            )

    async def crawl_queue_status(self) -> dict[str, int]:
        """Return persistent queue totals by state."""
        return await asyncio.to_thread(self._crawl_queue_status_sync)

    def _crawl_queue_status_sync(self) -> dict[str, int]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT status, COUNT(*) AS count FROM crawl_queue GROUP BY status"
            ).fetchall()
        totals = {"queued": 0, "processing": 0, "visited": 0, "failed": 0}
        totals.update({str(row["status"]): int(row["count"]) for row in rows})
        return totals

    def _count_sync(self, source_prefix: str | None) -> int:
        with self._connection() as connection:
            if source_prefix is None:
                row = connection.execute("SELECT COUNT(*) FROM memories").fetchone()
            else:
                row = connection.execute(
                    "SELECT COUNT(*) FROM memories "
                    "WHERE substr(source, 1, length(?)) = ?",
                    (source_prefix, source_prefix),
                ).fetchone()
        return int(row[0])

    def _history_sync(self, limit: int) -> list[sqlite3.Row]:
        now = datetime.now(timezone.utc).isoformat()
        with self._connection() as connection:
            return list(
                connection.execute(
                    "SELECT id, text, source, created_at FROM memories "
                    "WHERE expires_at IS NULL OR expires_at > ? "
                    "ORDER BY created_at DESC LIMIT ?",
                    (now, limit),
                )
            )

    def _read_search_candidates_sync(
        self, candidate_limit: int, fts_query: str = ""
    ) -> list[sqlite3.Row]:
        """Fetch a fixed candidate budget from recency and full-text matches."""
        bounded_candidate_limit = max(
            1, min(candidate_limit, SEARCH_CANDIDATE_LIMIT)
        )
        now = datetime.now(timezone.utc).isoformat()
        recent_limit = (
            max(1, bounded_candidate_limit - SEARCH_KEYWORD_CANDIDATE_LIMIT)
            if fts_query
            else bounded_candidate_limit
        )
        with self._connection() as connection:
            if fts_query:
                keyword_limit = min(
                    SEARCH_KEYWORD_CANDIDATE_LIMIT,
                    bounded_candidate_limit,
                )
                return list(
                    connection.execute(
                        "WITH keyword_hits AS ("
                        " SELECT memories_fts.rowid AS id FROM memories_fts "
                        " JOIN memories m ON m.id = memories_fts.rowid "
                        " WHERE memories_fts MATCH ? "
                        " AND (m.expires_at IS NULL OR m.expires_at > ?) "
                        " ORDER BY bm25(memories_fts) LIMIT ?"
                        "), recent AS ("
                        " SELECT id FROM memories WHERE "
                        "(expires_at IS NULL OR expires_at > ?) "
                        " ORDER BY created_at DESC, id DESC LIMIT ?"
                        "), candidate_ids AS ("
                        " SELECT id FROM keyword_hits UNION SELECT id FROM recent"
                        ") SELECT m.id, m.text, m.source, m.embedding, m.created_at "
                        "FROM candidate_ids c JOIN memories m ON m.id = c.id "
                        "WHERE m.expires_at IS NULL OR m.expires_at > ? "
                        "ORDER BY m.created_at DESC, m.id DESC LIMIT ?",
                        (
                            fts_query,
                            now,
                            keyword_limit,
                            now,
                            recent_limit,
                            now,
                            bounded_candidate_limit,
                        ),
                    )
                )
            return list(
                connection.execute(
                    "SELECT id, text, source, embedding, created_at FROM memories "
                    "WHERE expires_at IS NULL OR expires_at > ? "
                    "ORDER BY created_at DESC, id DESC LIMIT ?",
                    (now, bounded_candidate_limit),
                )
            )

    async def retention_report(
        self,
        retention_days: int,
        *,
        include_user_authored: bool = False,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        """Report old records eligible for deletion without deleting any data.

        By default only crawler-created memories (``crawl:`` sources) are
        eligible. Other sources are treated as user-authored or otherwise
        protected. Results are paginated and capped to avoid loading a large
        report into memory.
        """
        if retention_days < 1:
            raise ValueError("Retention must be at least one day.")
        bounded_limit = max(1, min(limit, RETENTION_REPORT_MAX_LIMIT))
        bounded_offset = max(0, offset)
        cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
        cutoff_text = cutoff.isoformat()
        records, eligible_count = await asyncio.to_thread(
            self._retention_report_sync,
            cutoff_text,
            include_user_authored,
            bounded_limit,
            bounded_offset,
        )
        return {
            "dry_run": True,
            "retention_days": retention_days,
            "cutoff": cutoff_text,
            "include_user_authored": include_user_authored,
            "eligible_count": eligible_count,
            "limit": bounded_limit,
            "offset": bounded_offset,
            "has_more": bounded_offset + len(records) < eligible_count,
            "records": records,
        }

    def _retention_report_sync(
        self,
        cutoff: str,
        include_user_authored: bool,
        limit: int,
        offset: int,
    ) -> tuple[list[dict[str, Any]], int]:
        with self._connection() as connection:
            authorship_filter = "" if include_user_authored else " AND user_authored = 0"
            eligible_count = int(
                connection.execute(
                    "SELECT COUNT(*) FROM memories WHERE created_at < ?"
                    + authorship_filter,
                    (cutoff,),
                ).fetchone()[0]
            )
            rows = connection.execute(
                "SELECT id, text, source, created_at FROM memories "
                "WHERE created_at < ?"
                + authorship_filter
                + " ORDER BY created_at ASC, id ASC LIMIT ? OFFSET ?",
                (cutoff, limit, offset),
            ).fetchall()
        records = [
            {
                "id": int(row["id"]),
                "source": str(row["source"]),
                "created_at": str(row["created_at"]),
                "preview": str(row["text"])[:300],
            }
            for row in rows
        ]
        return records, eligible_count

    async def prune(
        self, retention_days: int, *, include_user_authored: bool = False
    ) -> int:
        """Delete expired memories only when explicitly invoked.

        The default policy protects every non-crawler memory. Set
        ``include_user_authored=True`` only after an explicit user confirmation.
        """
        if retention_days < 1:
            raise ValueError("Retention must be at least one day.")
        cutoff = (datetime.now(timezone.utc) - timedelta(days=retention_days)).isoformat()
        self.logger.warning(
            "Memory prune requested: retention_days=%d include_user_authored=%s cutoff=%s",
            retention_days,
            include_user_authored,
            cutoff,
        )
        async with self._lock:
            deleted_count = await asyncio.to_thread(
                self._prune_sync, cutoff, include_user_authored
            )
        self.logger.warning(
            "Memory prune completed: deleted=%d retention_days=%d "
            "include_user_authored=%s cutoff=%s",
            deleted_count,
            retention_days,
            include_user_authored,
            cutoff,
        )
        return deleted_count

    def _prune_sync(self, cutoff: str, include_user_authored: bool) -> int:
        authorship_filter = "" if include_user_authored else " AND user_authored = 0"
        with self._connection() as connection:
            cursor = connection.execute(
                "DELETE FROM memories WHERE created_at < ?" + authorship_filter,
                (cutoff,),
            )
            connection.execute(
                "DELETE FROM technical_documents WHERE NOT EXISTS ("
                "SELECT 1 FROM memories WHERE memories.document_url = technical_documents.url"
                ")"
            )
            return max(0, cursor.rowcount)


def _cosine_similarity(left: list[float], right: list[float]) -> float | None:
    """Compute bounded cosine similarity for equal-length finite vectors."""
    if len(left) != len(right) or not left:
        return None
    left_array = np.asarray(left, dtype=np.float32)
    right_array = np.asarray(right, dtype=np.float32)
    denominator = float(np.linalg.norm(left_array) * np.linalg.norm(right_array))
    if denominator == 0 or not math.isfinite(denominator):
        return None
    value = float(np.dot(left_array, right_array) / denominator)
    return max(-1.0, min(1.0, value)) if math.isfinite(value) else None


def _memory_category(source: str) -> str:
    """Map the stable memory source prefix to a bounded local category."""
    if source.startswith("techdoc:"):
        return "technical_documentation"
    if source.startswith("crawl:"):
        return "crawled_content"
    if source == "user":
        return "user_note"
    return "other"


def prepare_pending_memory_chunks(text: str) -> list[str]:
    """Clean saved text and split it into at most four bounded local chunks."""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return []
    blocks = [
        block.strip()
        for block in re.split(r"\n\s*\n", normalized)
        if block.strip()
        and (
            block.strip().startswith(("```", "~~~"))
            or not _PENDING_BOILERPLATE_PATTERN.search(block)
        )
    ]
    if not blocks:
        return []

    chunks: list[str] = []
    current = ""
    for block in blocks:
        remaining = block
        while len(remaining) > PENDING_MEMORY_CHUNK_CHARS:
            boundary = max(
                remaining.rfind("\n", 0, PENDING_MEMORY_CHUNK_CHARS + 1),
                remaining.rfind(" ", 0, PENDING_MEMORY_CHUNK_CHARS + 1),
            )
            if boundary < PENDING_MEMORY_CHUNK_CHARS // 2:
                boundary = PENDING_MEMORY_CHUNK_CHARS
            piece = remaining[:boundary].strip()
            if piece:
                combined = f"{current}\n\n{piece}" if current else piece
                if len(combined) <= PENDING_MEMORY_CHUNK_CHARS:
                    current = combined
                else:
                    chunks.append(current)
                    current = piece
            remaining = remaining[boundary:].strip()

        if remaining:
            combined = f"{current}\n\n{remaining}" if current else remaining
            if len(combined) <= PENDING_MEMORY_CHUNK_CHARS:
                current = combined
            else:
                chunks.append(current)
                current = remaining
    if current:
        chunks.append(current)
    if len(chunks) > 4 or any(
        len(chunk) > PENDING_MEMORY_CHUNK_CHARS for chunk in chunks
    ):
        raise ValueError("Pending memory could not be split within configured limits.")
    return chunks


def _search_tokens(text: str) -> list[str]:
    """Extract a small, deterministic token set that preserves code identifiers."""
    tokens: list[str] = []
    seen: set[str] = set()
    for match in _SEARCH_TOKEN_PATTERN.findall(text.casefold()):
        token = match.strip("._-+#")
        if (
            len(token) < 2
            or token in _SEARCH_STOP_WORDS
            or token in seen
        ):
            continue
        seen.add(token)
        tokens.append(token)
        if len(tokens) >= 16:
            break
    return tokens


def _lexical_score(text: str, query_tokens: list[str]) -> float:
    """Score overlap while strongly rewarding exact technical identifiers."""
    if not query_tokens:
        return 0.0
    document_tokens = set(_SEARCH_TOKEN_PATTERN.findall(text.casefold()))
    matching_tokens = [token for token in query_tokens if token in document_tokens]
    overlap = len(matching_tokens) / len(query_tokens)
    if any(
        ("_" in token or any(character.isdigit() for character in token) or len(token) >= 10)
        for token in matching_tokens
    ):
        overlap = max(overlap, 0.8)
    return min(1.0, overlap)


def _is_low_quality_memory(text: str, source: str) -> bool:
    """Exclude short or noisy memories and non-technical crawler chunks."""
    normalized = text.strip()
    if len(normalized) < 40 or _CRAWLER_NOISE_PATTERN.search(normalized):
        return True
    if not source.startswith(("crawl:", "techdoc:")):
        return False
    if len(normalized) < 80:
        return True
    unique_terms = {
        token
        for token in _SEARCH_TOKEN_PATTERN.findall(normalized.casefold())
        if len(token) > 1 and token not in _SEARCH_STOP_WORDS
    }
    return len(unique_terms) < 8 or not _TECHNICAL_CONTENT_PATTERN.search(normalized)


def _record(row: sqlite3.Row, similarity: float | None) -> MemoryRecord:
    """Map a SQLite row to the public record type."""
    return MemoryRecord(
        id=int(row["id"]),
        text=str(row["text"]),
        source=str(row["source"]),
        created_at=str(row["created_at"]),
        similarity=similarity,
    )
