"""Application settings and logging configuration for OmniMind Local."""

from __future__ import annotations

import logging
import ipaddress
import os
import sys
from dataclasses import dataclass
from logging.handlers import RotatingFileHandler
from pathlib import Path
from urllib.parse import urlsplit

import certifi


BASE_DIR = (
    Path(sys.executable).resolve().parent
    if getattr(sys, "frozen", False)
    else Path(__file__).resolve().parent
)
RESOURCE_DIR = Path(getattr(sys, "_MEIPASS", BASE_DIR))
_LOCAL_APP_DATA = Path(
    os.getenv("LOCALAPPDATA", str(Path.home() / "AppData" / "Local"))
)
DATA_DIR = (
    _LOCAL_APP_DATA / "OmniMind Local"
    if getattr(sys, "frozen", False)
    else BASE_DIR
)
LANGUAGE_PATH = DATA_DIR / "language.txt"
_configured_ca_file = os.getenv("SSL_CERT_FILE", "").strip()
if not _configured_ca_file or not Path(_configured_ca_file).is_file():
    os.environ["SSL_CERT_FILE"] = certifi.where()


@dataclass(frozen=True, slots=True)
class Settings:
    """Validated runtime settings, loaded from environment variables."""

    app_name: str = "OmniMind Local"
    language: str = "en"
    host: str = "127.0.0.1"
    port: int = 8765
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_embedding_model: str = "nomic-embed-text"
    ollama_chat_model: str = "qwen2.5"
    ollama_keep_alive: str = "1m"
    ollama_num_ctx: int = 2048
    ollama_num_predict: int = 512
    ollama_num_thread: int = 4
    update_manifest_url: str = ""
    update_public_key: str = ""
    database_path: Path = DATA_DIR / "data" / "omnimind.sqlite3"
    log_path: Path = DATA_DIR / "logs" / "omnimind.log"
    allowed_crawl_hosts: tuple[str, ...] = ()
    crawl_interval_seconds: float = 5.0
    crawl_timeout_seconds: float = 15.0
    crawl_max_response_bytes: int = 1_000_000
    crawl_max_retries: int = 2
    crawl_max_concurrent_requests: int = 2
    crawl_queue_max_pending: int = 100
    metrics_interval_seconds: float = 2.0
    memory_retention_days: int = 180
    technical_learning_enabled: bool = True
    technical_learning_relevance_threshold: float = 0.35
    technical_learning_max_records: int = 500
    technical_learning_max_chunks_per_page: int = 4
    technical_learning_chunk_chars: int = 1000
    technical_learning_queue_max_pending: int = 50
    technical_learning_ttl_days: int = 90
    technical_learning_retry_seconds: int = 300
    cpu_warning_percent: float = 85.0
    memory_warning_percent: float = 85.0
    disk_warning_percent: float = 90.0

    @classmethod
    def from_env(cls) -> Settings:
        """Build settings from the process environment, including a local .env file."""
        _load_dotenv(DATA_DIR / ".env")
        if DATA_DIR != BASE_DIR:
            _load_dotenv(BASE_DIR / ".env")

        raw_hosts = os.getenv("OMNIMIND_ALLOWED_HOSTS", "")
        hosts = tuple(
            sorted(
                {
                    item.strip().lower().rstrip(".")
                    for item in raw_hosts.split(",")
                    if item.strip()
                }
            )
        )
        settings = cls(
            language=_app_language(),
            host=os.getenv("OMNIMIND_HOST", "127.0.0.1"),
            port=_int_env("OMNIMIND_PORT", 8765),
            ollama_base_url=os.getenv(
                "OMNIMIND_OLLAMA_URL", "http://127.0.0.1:11434"
            ).rstrip("/"),
            ollama_embedding_model=os.getenv(
                "OMNIMIND_EMBEDDING_MODEL", "nomic-embed-text"
            ),
            ollama_chat_model=os.getenv("OMNIMIND_CHAT_MODEL", "qwen2.5").strip(),
            ollama_keep_alive=os.getenv("OMNIMIND_OLLAMA_KEEP_ALIVE", "1m").strip(),
            ollama_num_ctx=_int_env("OMNIMIND_OLLAMA_NUM_CTX", 2048),
            ollama_num_predict=_int_env("OMNIMIND_OLLAMA_NUM_PREDICT", 512),
            ollama_num_thread=_int_env("OMNIMIND_OLLAMA_NUM_THREAD", 4),
            update_manifest_url=os.getenv("OMNIMIND_UPDATE_URL", "").strip(),
            update_public_key=os.getenv("OMNIMIND_UPDATE_PUBLIC_KEY", "").strip(),
            database_path=Path(
                os.getenv(
                    "OMNIMIND_DATABASE",
                    str(DATA_DIR / "data" / "omnimind.sqlite3"),
                )
            ).expanduser(),
            log_path=Path(
                os.getenv("OMNIMIND_LOG", str(DATA_DIR / "logs" / "omnimind.log"))
            ).expanduser(),
            allowed_crawl_hosts=hosts,
            crawl_interval_seconds=_float_env("OMNIMIND_CRAWL_INTERVAL", 5.0),
            crawl_timeout_seconds=_float_env("OMNIMIND_CRAWL_TIMEOUT", 15.0),
            crawl_max_response_bytes=_int_env("OMNIMIND_CRAWL_MAX_BYTES", 1_000_000),
            crawl_max_retries=_int_env("OMNIMIND_CRAWL_MAX_RETRIES", 2),
            crawl_max_concurrent_requests=_int_env(
                "OMNIMIND_CRAWL_MAX_CONCURRENT", 2
            ),
            crawl_queue_max_pending=_int_env("OMNIMIND_CRAWL_QUEUE_MAX_PENDING", 100),
            metrics_interval_seconds=_float_env("OMNIMIND_METRICS_INTERVAL", 2.0),
            memory_retention_days=_int_env("OMNIMIND_MEMORY_RETENTION_DAYS", 180),
            technical_learning_enabled=_bool_env(
                "OMNIMIND_TECHNICAL_LEARNING_ENABLED", True
            ),
            technical_learning_relevance_threshold=_float_env(
                "OMNIMIND_TECHNICAL_RELEVANCE_THRESHOLD", 0.35
            ),
            technical_learning_max_records=_int_env(
                "OMNIMIND_TECHNICAL_MAX_RECORDS", 500
            ),
            technical_learning_max_chunks_per_page=_int_env(
                "OMNIMIND_TECHNICAL_MAX_CHUNKS_PER_PAGE", 4
            ),
            technical_learning_chunk_chars=_int_env(
                "OMNIMIND_TECHNICAL_CHUNK_CHARS", 1000
            ),
            technical_learning_queue_max_pending=_int_env(
                "OMNIMIND_TECHNICAL_QUEUE_MAX_PENDING", 50
            ),
            technical_learning_ttl_days=_int_env(
                "OMNIMIND_TECHNICAL_TTL_DAYS", 90
            ),
            technical_learning_retry_seconds=_int_env(
                "OMNIMIND_TECHNICAL_RETRY_SECONDS", 300
            ),
            cpu_warning_percent=_float_env("OMNIMIND_CPU_WARNING_PERCENT", 85.0),
            memory_warning_percent=_float_env("OMNIMIND_MEMORY_WARNING_PERCENT", 85.0),
            disk_warning_percent=_float_env("OMNIMIND_DISK_WARNING_PERCENT", 90.0),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        """Reject unsafe or internally inconsistent settings."""
        if self.host not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("The dashboard must bind to loopback only.")
        if self.language not in {"en", "ka", "ru"}:
            raise ValueError("The application language must be en, ka, or ru.")
        if self.update_manifest_url:
            update_url = urlsplit(self.update_manifest_url)
            if (
                update_url.scheme != "https"
                or not update_url.hostname
                or update_url.username is not None
                or update_url.password is not None
            ):
                raise ValueError("OMNIMIND_UPDATE_URL must be a credential-free HTTPS URL.")
        if not 1 <= self.port <= 65535:
            raise ValueError("OMNIMIND_PORT must be between 1 and 65535.")
        parsed = urlsplit(self.ollama_base_url)
        if (
            parsed.scheme not in {"http", "https"}
            or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}
            or parsed.username is not None
            or parsed.password is not None
        ):
            raise ValueError("Ollama must use a loopback URL without credentials.")
        if not self.ollama_embedding_model.strip():
            raise ValueError("OMNIMIND_EMBEDDING_MODEL cannot be empty.")
        if not self.ollama_chat_model:
            raise ValueError("OMNIMIND_CHAT_MODEL cannot be empty.")
        if not self.ollama_keep_alive:
            raise ValueError("OMNIMIND_OLLAMA_KEEP_ALIVE cannot be empty.")
        if not 512 <= self.ollama_num_ctx <= 4096:
            raise ValueError("OMNIMIND_OLLAMA_NUM_CTX must be between 512 and 4096.")
        if not 1 <= self.ollama_num_thread <= 8:
            raise ValueError("OMNIMIND_OLLAMA_NUM_THREAD must be between 1 and 8.")
        if not 64 <= self.ollama_num_predict <= 2048:
            raise ValueError("OMNIMIND_OLLAMA_NUM_PREDICT must be between 64 and 2048.")
        if any(not _valid_hostname(host) for host in self.allowed_crawl_hosts):
            raise ValueError("OMNIMIND_ALLOWED_HOSTS contains an invalid host.")
        if not 1.0 <= self.crawl_interval_seconds <= 3600.0:
            raise ValueError("The crawl interval must be between 1 and 3600 seconds.")
        if not 0 < self.crawl_timeout_seconds <= 300:
            raise ValueError("Crawler timeout must be between 0 and 300 seconds.")
        if not 0 < self.crawl_max_response_bytes <= 5_000_000:
            raise ValueError("Crawler response size must be between 1 and 5000000 bytes.")
        if not 0 <= self.crawl_max_retries <= 5:
            raise ValueError("OMNIMIND_CRAWL_MAX_RETRIES must be between 0 and 5.")
        if not 1 <= self.crawl_max_concurrent_requests <= 4:
            raise ValueError("OMNIMIND_CRAWL_MAX_CONCURRENT must be between 1 and 4.")
        if not 1 <= self.crawl_queue_max_pending <= 500:
            raise ValueError("OMNIMIND_CRAWL_QUEUE_MAX_PENDING must be between 1 and 500.")
        if not 1 <= self.technical_learning_queue_max_pending <= 100:
            raise ValueError("Technical learning queue capacity must be between 1 and 100.")
        if self.metrics_interval_seconds < 0.5:
            raise ValueError("The metrics interval cannot be less than 0.5 seconds.")
        if self.memory_retention_days < 1:
            raise ValueError("Memory retention must be at least one day.")
        if not 0 <= self.technical_learning_relevance_threshold <= 1:
            raise ValueError("Technical relevance threshold must be between 0 and 1.")
        if not 1 <= self.technical_learning_max_records <= 2000:
            raise ValueError("Technical memory capacity must be between 1 and 2000.")
        if not 1 <= self.technical_learning_max_chunks_per_page <= 10:
            raise ValueError("Technical chunks per page must be between 1 and 10.")
        if not 256 <= self.technical_learning_chunk_chars <= 5000:
            raise ValueError("Technical chunk size must be between 256 and 5000.")
        if not 1 <= self.technical_learning_ttl_days <= 3650:
            raise ValueError("Technical document TTL must be between 1 and 3650 days.")
        if self.technical_learning_retry_seconds < 1:
            raise ValueError("Technical retry delay must be positive.")
        for name, value in (
            ("CPU", self.cpu_warning_percent),
            ("memory", self.memory_warning_percent),
            ("disk", self.disk_warning_percent),
        ):
            if not 0 < value <= 100:
                raise ValueError(f"{name} warning percentage must be in (0, 100].")


def _app_language() -> str:
    configured = os.getenv("OMNIMIND_LANGUAGE", "").strip().lower()
    if configured:
        return configured
    try:
        stored = LANGUAGE_PATH.read_text(encoding="utf-8").strip().lower()
    except OSError:
        return "en"
    return stored or "en"


def _load_dotenv(path: Path) -> None:
    """Load a .env file without requiring python-dotenv; keep explicit environment variables authoritative."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, raw_value = stripped.split("=", 1)
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export ") :].strip()
        if key in os.environ:
            continue
        value = raw_value.strip().strip('"').strip("'")
        os.environ[key] = value


def _valid_hostname(host: str) -> bool:
    """Accept DNS names only; IP literals and wildcard patterns are not allowed."""
    if not host or len(host) > 253 or "*" in host or ":" in host:
        return False
    try:
        ipaddress.ip_address(host)
    except ValueError:
        pass
    else:
        return False
    if host in {"localhost", "localhost.localdomain"}:
        return False
    return all(
        0 < len(label) <= 63
        and label[0].isalnum()
        and label[-1].isalnum()
        and all(character.isalnum() or character == "-" for character in label)
        for label in host.split(".")
    )


def _int_env(name: str, default: int) -> int:
    """Parse an integer environment variable with a useful error message."""
    try:
        return int(os.getenv(name, str(default)))
    except ValueError as error:
        raise ValueError(f"{name} must be an integer.") from error


def _float_env(name: str, default: float) -> float:
    """Parse a floating-point environment variable with a useful error message."""
    try:
        return float(os.getenv(name, str(default)))
    except ValueError as error:
        raise ValueError(f"{name} must be a number.") from error


def _bool_env(name: str, default: bool) -> bool:
    """Parse a boolean environment variable using common explicit values."""
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    normalized = raw_value.strip().casefold()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value.")


class _ColorFormatter(logging.Formatter):
    """Apply concise ANSI colors to console log levels."""

    _COLORS = {
        logging.DEBUG: "\033[90m",
        logging.INFO: "\033[36m",
        logging.WARNING: "\033[33m",
        logging.ERROR: "\033[31m",
        logging.CRITICAL: "\033[1;31m",
    }
    _RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self._COLORS.get(record.levelno, "")
        rendered = super().format(record)
        return f"{color}{rendered}{self._RESET}" if color else rendered


def configure_logging(settings: Settings) -> logging.Logger:
    """Configure rotating file and colorful console handlers once."""
    settings.log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("omnimind")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    file_handler = RotatingFileHandler(
        settings.log_path, maxBytes=5_000_000, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    if sys.stderr is not None:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(_ColorFormatter(formatter._fmt, formatter.datefmt))
        logger.addHandler(console_handler)
    return logger
