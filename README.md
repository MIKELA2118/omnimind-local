# OmniMind Local

A local-first AI dashboard for aggregate hardware telemetry, SQLite-backed semantic memory, local Ollama chat, and deliberately scoped web fetching. The server binds to loopback by default. Crawling is disabled until exact DNS hostnames are configured in `OMNIMIND_ALLOWED_HOSTS`.

## Requirements

- Python 3.10 or newer
- Ollama installed locally for embeddings
- The `nomic-embed-text` model (or another Ollama embedding model)
- A local chat model configured by `OMNIMIND_CHAT_MODEL` (default: `qwen2.5`)

Install Ollama and the embedding model once, then install the Python packages.
OmniMind automatically starts the Ollama service whenever it is not already
running and downloads the configured chat model if needed.

```powershell
ollama pull nomic-embed-text
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The requirements include the native-window wrapper, the pure-Python `wsproto`
backend, and Uvicorn's `websockets` backend. On Windows, pywebview uses the
Microsoft Edge WebView2 runtime; Windows 11 includes it, while some Windows 10
installations may need the WebView2 Runtime installed separately. To install
just the WebSocket dependencies while requiring wheels (no source builds), run:

```powershell
python -m pip install --only-binary=:all: websockets wsproto
```

On macOS/Linux, activate the environment with `source .venv/bin/activate`.

## Desktop app

Copy `.env.example` to `.env` and set `OMNIMIND_ALLOWED_HOSTS` to exact DNS hosts you are authorized to fetch. `aura.ge` also permits its subdomains for its grammar resources; wildcard entries are not accepted. The launcher reads `.env` without requiring `python-dotenv`.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
& ".\OmniMind Local.bat"
```

The branded launcher verifies its Python dependencies, starts FastAPI quietly
in the background, and opens a native OmniMind Local window with its own
taskbar entry and icon. Closing the window stops the local server. If Ollama is
installed but not running, OmniMind starts it automatically; if Ollama is
missing or cannot start, the dashboard still opens and shows a friendly status
message. Startup errors appear in a Windows dialog and are written to the
configured application log. If Ollama is installed in a non-standard location,
set `OMNIMIND_OLLAMA_EXECUTABLE` to its executable path in `.env`. Run
`powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\install_shortcut.ps1`
once to add branded shortcuts to the desktop and Start menu.

Chat and embeddings are sent only to the configured local Ollama server. On
startup, OmniMind checks whether the configured chat model is installed and
automatically pulls it from Ollama if missing; the first download requires an
internet connection. Pull progress and failures are shown in the dashboard
model status.

## Build a Windows installer

The build creates a standalone, single-file Windows setup executable using
Inno Setup 6 and the existing `installer\OmniMind.iss` script. It installs the
application per user, registers an uninstaller, adds Start-menu shortcuts, and
offers a desktop shortcut. Ollama models and the WebView2 Runtime are not
bundled. Install Inno Setup 6 and the project Python dependencies, then run:

```powershell
.\.venv\Scripts\python.exe -m pip install pyinstaller
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\build_windows.ps1
```

The script builds and verifies the PyInstaller runtime bundle in the ignored
`build\package` staging directory, compiles the Inno Setup installer, and checks
that the versioned setup file exists at
`dist\installer\OmniMind-Local-Setup-<version>.exe`. Staging separately avoids
overwriting a previously built or running app in `dist`. The build reads the
version from `version.py`; if Inno Setup is installed outside its standard
location, set `INNO_SETUP_COMPILER` to the full path of `ISCC.exe`. The
installer offers English, Georgian, and Russian setup languages, saves the
selected app language under `%LOCALAPPDATA%\OmniMind Local\language.txt`, and
offers to open Ollama's official Windows download page after setup. Packaged
user data and logs are kept under `%LOCALAPPDATA%\OmniMind Local`, not inside
the installed program folder. The generated installer is not Authenticode
signed; sign the release with the publisher's code-signing certificate before
public distribution to avoid Windows publisher warnings.

## Signed application updates

Startup update checks and in-app installation are disabled by default. To
enable them, publish an HTTPS JSON manifest and set both
`OMNIMIND_UPDATE_URL` and `OMNIMIND_UPDATE_PUBLIC_KEY` in the user's `.env`
file. The public key must be the base64-encoded 32-byte Ed25519 public key.
Never put the private signing key in the application or on the user's machine.
The manifest format is:

```json
{
  "version": "1.0.1",
  "installer_url": "https://updates.example.com/OmniMind-Local-Setup-1.0.1.exe",
  "sha256": "<64 lowercase hexadecimal characters>",
  "signature": "<base64 Ed25519 signature>",
  "release_notes": "Optional release notes"
}
```

The Ed25519 signature is over the UTF-8 bytes of
`version + "\n" + installer_url + "\n" + sha256` (with the digest lowercase).
The app accepts only HTTPS manifest and installer URLs, verifies the signed
metadata before showing the update, then checks the installer SHA-256 before
launching it. Updates run the normal Inno Setup package silently and relaunch
the app. The hosted manifest URL and trusted public key are deployment values
and must be supplied by the release operator; no Render endpoint or signing
key is configured in this source tree.
For development, `.\.venv\Scripts\pythonw.exe launch.py` opens the same native
desktop app without a console; `python app.py` runs the API directly for
backend debugging.

Dashboard markup lives in `dashboard.html`; its styles and client-side behavior
are served separately from `static/dashboard.css` and `static/dashboard.js`.

Ollama requests use `OMNIMIND_OLLAMA_KEEP_ALIVE` (default `1m`), `OMNIMIND_OLLAMA_NUM_CTX` (default `2048`, maximum `4096`), and `OMNIMIND_OLLAMA_NUM_THREAD` (default `4`, maximum `8`) to limit model context and CPU concurrency. `OMNIMIND_OLLAMA_NUM_PREDICT` (default `512`, maximum `2048`) bounds generated tokens. These settings reduce resource use but cannot guarantee a total 8 GB system or model RAM ceiling; actual memory also depends on the selected model and other running applications. Chat uses a 300-second Ollama read timeout. The dashboard consumes WebSocket status updates and reconnects automatically. Crawl response size, timeout, retry count, pace, queue capacity, and monitoring frequency are configurable in `.env`.

## Technical documentation learning

Chat first applies a lightweight technical-intent check. Georgian grammar
questions about Aura.ge are included in the curated learning catalog. The
pipeline follows at most 10 discovered Aura.ge grammar-related links per
server process.
Greetings and ordinary non-technical conversation skip both SQLite vector
retrieval and documentation learning. Technical prompts search at most 100
records total (up to 50 recent
and 50 SQLite FTS5 matches), blending exact identifier/error-code matches with
local embedding similarity. Retrieval still includes a bounded context/history
window. When a technical query has no
sufficiently relevant local vector result, OmniMind schedules (without delaying
the answer) **one** matching page from a fixed catalog: the Python
tutorial, FastAPI tutorial, MDN JavaScript guide, TypeScript documentation,
React Learn, or Supabase Getting Started. Matching uses fixed keywords; the model cannot create URLs,
choose arbitrary hosts, follow redirects, or expand the catalog at runtime.
The selected hostname must be covered by `OMNIMIND_ALLOWED_HOSTS`.

`POST /api/learn/urls` accepts a bounded JSON object such as
`{"urls":[{"url":"https://aura.ge/"}]}`. URLs are validated against the
allowlist before admission to a single-worker background queue; queue status,
capacity, and completion/failure counts are available through technical-learning
status. Existing `POST /api/learn/url` submits one URL to the same queue.

`POST /api/learn/process-pending` and the dashboard's **Process Pending Queue**
button process pending records and crawl URLs already stored in SQLite. It
cleans saved text, splits it into bounded chunks, creates local Ollama
embeddings, and fetches queued URLs sequentially through the existing
allowlist, timeout, response-size, retry, and per-host rate-limit protections.
It does not search for or discover additional URLs. Progress, current record or
URL, and completed/failed counts are included in `/ws` status snapshots.
Processing uses a two-record SQLite read batch with at most four 5,000-character
chunks per record; each fetched page's chunks are embedded and stored atomically.
An unavailable embedding service leaves the current URL queued and pending
memory records retryable.

The shared crawler removes hidden content, page chrome, navigation, cookie
notices, advertising, and link-heavy menus before extracting documentation. It
returns semantic document blocks, inline code, and fenced code blocks as
Markdown. Retrieval excludes short, noisy, and non-technical chunks from
crawler-created memories. Each fetch is limited by the existing crawler timeout
and response size. Requests are serialized by allowlisted host root, separated
by a polite default five-second delay with at most two simultaneous requests
across distinct hosts, and transient network, rate-limit, timeout, and server errors use bounded exponential backoff; permanent errors such as 404 are
reported without retrying. Markdown is split into bounded chunks (default 1000
characters, maximum 4 chunks per page), embedded
locally with Ollama using one concurrent embedding request, and committed
atomically to SQLite. Technical chunks have their own hard storage cap (default
500); exceeding it skips further indexing
without deleting older or user-authored memories. Indexed documentation URLs
are tracked with source-version metadata, indexing timestamps, and a configurable
freshness TTL (default 90 days, `OMNIMIND_TECHNICAL_TTL_DAYS`). Fenced code is
kept intact or split only at top-level function/class boundaries; a single
oversized logical code block is rejected rather than cut arbitrarily. Expired
documents are reported at `GET /api/technical-learning/freshness` and refreshed
when a matching query selects them again. Stale documentation is not deleted
automatically.

Operators can submit one explicit page using
`POST /api/learn/url` with `{"url":"https://docs.python.org/3/tutorial/"}`.
The URL must use HTTP(S), contain no credentials or fragment, and match an
exact hostname in `OMNIMIND_ALLOWED_HOSTS`; redirects remain disabled. The
endpoint returns `202 Accepted` and schedules fetch, cleaning, chunking,
embedding, and indexing in the background. Only one technical-learning task
runs at a time.

Technical learning is enabled by default but can be disabled with
`OMNIMIND_TECHNICAL_LEARNING_ENABLED=false`. Tune the trigger threshold,
per-page chunk limits, record cap, and retry cooldown with the
`OMNIMIND_TECHNICAL_*` settings in `.env`. `GET /api/technical-learning/status`
and the dashboard show catalog allowlist availability and index usage. Chat
responses include a `knowledge_learning` field indicating whether ingestion was
scheduled. A single learning job may run at once; newly indexed content is
available to subsequent RAG searches. This is query-triggered learning, not an
endless crawler; the existing background crawler remains explicitly user-started.

## In-tree plugin registry

Plugins are trusted Python modules committed inside the `plugins/` package and
registered explicitly in `plugins/__init__.py`. The registry validates
required metadata, stable IDs, unique names and endpoints, declared
capabilities, callable HTTP endpoints, and the required
`/api/plugins/{plugin-id}/...` route prefix before mounting routers. Validation
also runs at application startup. `GET /api/plugins` lists the registered
metadata and endpoint declarations; the built-in text tools example is
`POST /api/plugins/text-tools/analyze`.

To add a plugin safely:

1. Add a reviewed Python module under `plugins/` with an `APIRouter` whose
   prefix is exactly `/api/plugins/<stable-id>` and define bounded request
   models for any user-provided input.
2. Export a `PluginDefinition` with a lowercase stable ID, unique display name,
   version, description, one or more capability identifiers, and the router.
3. Import that definition and call `plugin_registry.register(...)` explicitly
   in `plugins/__init__.py`.
4. Start OmniMind and confirm the metadata appears at `GET /api/plugins`;
   startup validation rejects duplicate or malformed registrations.

Plugins are not discovered from a writable extensions directory, hot-reloaded,
or generated/executed by the LLM. New plugin code must be reviewed and added
to the in-tree registry by an operator before the server starts.

## API

Send a crawl request to `POST /api/crawl` with a JSON body containing `url`. Set
`store` to `true` to save the extracted page text (up to 20,000 characters) in
local memory; it defaults to `false`.

```json
{
  "url": "https://example.com",
  "store": true
}
```

The response contains crawl metadata and extracted content under `result`, and
the saved memory record under `memory` when storage is enabled.

Memory retention is never applied automatically. `GET
/api/memory/retention/report` returns a paginated dry-run report using
`OMNIMIND_MEMORY_RETENTION_DAYS`; it only lists records and does not delete
anything. By default, only crawler-authored records are eligible. User-created
memories are marked as protected regardless of their source label. After
reviewing the report, `POST /api/memory/retention/prune` with
`{"confirmed":true}` explicitly deletes eligible expired crawler memories.
Including user-authored records requires the separate explicit
`"include_user_authored":true` override. Prune requests and deletion counts are
written to the application log.

Use `POST /api/chat` with `{"prompt":"..."}` for a local Ollama response grounded
in retrieved SQLite memories. Optionally include a `history` array of up to 12
`{"role":"user"|"assistant","content":"..."}` turns for conversation continuity.
The response includes memory source references. `GET /api/proactive-greeting`
selects a recent learned memory and asks the local model to open a Georgian
conversation with Guga; if no memory exists, it starts without inventing a fact.
The dashboard requests this greeting when it loads and keeps up to 40 chat turns
in that browser's local storage.

Start a bounded, user-requested background crawl batch with
`POST /api/learning/start`, using `{"seed_url":"https://allowed.example/","max_pages":5}`.
The maximum is 10 pages per job. The seed and discovered links must match exact
hosts in `OMNIMIND_ALLOWED_HOSTS`. View progress at `GET /api/learning/status`
and stop a job with `POST /api/learning/stop`. To opt into a continuous daemon,
send `"continuous":true` with the same request (the dashboard has a matching
toggle). It drains the persistent, deduplicated SQLite queue and discovers more
allowlisted URLs until stopped. Continuous mode is never resumed automatically
after a server restart. The pending queue is capped by
`OMNIMIND_CRAWL_QUEUE_MAX_PENDING` (default 100). Page text is stored in
overlapping chunks so RAG can retrieve relevant sections.

## Behavior and boundaries

- System metrics are aggregate CPU, memory, disk, and network byte counters. Process lists and active connection details are not collected.
- Ollama embedding requests are sent only to a loopback Ollama endpoint. The app does not send data to paid APIs.
- Web fetching accepts HTTP or HTTPS URLs on exact allowlisted DNS hosts, does not follow redirects, applies per-host rate limits, and caps response size. The allowlist is empty by default.
- Background crawling runs only after a user starts it, follows allowlisted hosts (with subdomain matching only for `aura.ge`), waits at least five seconds between requests by default, exponentially backs off after transient errors, caps the pending SQLite queue (`OMNIMIND_CRAWL_QUEUE_MAX_PENDING`, default 100), and tracks queued, visited, and failed URLs. Continuous mode remains opt-in and stops on request or server shutdown. There is no shell, script execution, or self-modifying code.
- The server does not let the model write and execute plugins or install packages. New capabilities should be installed and reviewed by the operator before enabling them.
- The `/ws` WebSocket publishes dashboard updates and does not accept commands.
One background publisher collects system, memory, crawl-queue, learning, and
model status once per interval and broadcasts the latest snapshot to clients.
Each client has a one-message queue; when a client falls behind, stale updates
are replaced instead of accumulating in memory. Transient collection failures
are logged, and the publisher keeps the last known value for the affected
section while continuing to serve clients.

## Configuration

See `.env.example` for environment variables. `OMNIMIND_DATABASE` and `OMNIMIND_LOG` may be absolute paths or paths relative to the project directory. The SQLite database is created automatically.
