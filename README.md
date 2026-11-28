# Long-term memory service for a personal assistant

An HTTP service that receives a person's history one message at a time and has to **remember**, **update**, **recall or forget**, and **act** through tools. It implements the five endpoints of the event contract (`SPEC.md`) and ships a separate local product UI. The design is explained in [MEMORY.md](MEMORY.md); this file is about running it and what it depends on.

## Run it

**Official way (Docker, port 8000):**
```bash
cp .env.example .env        # Windows: copy .env.example .env  - then paste the key; .env is git-ignored
docker compose up --build   # service "memory" on http://localhost:8000
python scripts/check_submission.py . --live http://localhost:8000
```
The first build downloads the local embedding model into the image (`build.sh`, needs internet); the running container is offline and reaches only the model gateway and the mock world.

**Without Docker (Windows use `python`, macOS `python3`):**
```bash
python server.py                                   # official service on :8000, reads .env
python -m unittest discover -s tests               # 213 unit tests, no network or key needed
python -m world.server --port 8800 --persona data/dev/persona_a --date 2026-11-03     # mock world
python -m harness.run --system http://localhost:8000 --data data/dev/persona_a --world auto --judge offline --out runs/try1
```
Optional semantic search outside Docker: `pip install fastembed` (the model downloads once). Without it the service uses keyword search only.

**Product UI** (not part of the graded service; own storage, separate from the evaluation database):
```bash
python frontend/serve.py     # http://127.0.0.1:8080  (storage: $HUSH_PRODUCT_DIR or <tempdir>/hush_product)
```
Projects with isolated memory, persistent conversations, Tell / Ask / Do with receipts, answer details (evidence, retrieval steps, measured model usage), memory timeline, forget with verification. `Do` needs the mock world's tools URL.

## Endpoints (unchanged from SPEC.md)
`GET /health`, `POST /reset`, `POST /message`, `POST /answer` (`{answer, evidence, trace}`), `POST /act`. `server.py` is the organizers' wrapper; the implementation is `memory.py` -> `backend/source/`.

## Configuration
| Variable | Meaning |
|---|---|
| `OPENAI_BASE_URL`, `OPENAI_API_KEY`, `MODEL` | the model gateway; the only model used is `deepseek/deepseek-v4.1-flash` |
| `MEMORY_DB` | SQLite file of the service (default: system temp dir) |
| `MEMORY_EMBEDDINGS=off` | switch semantic search off; `EMBED_CACHE`, `EMBED_MODEL`, `SEMANTIC_MIN` tune it |
| `HUSH_PRODUCT_DIR`, `UI_PORT` | product UI storage and port |

## Layout
```
memory.py, server.py          official entry points
backend/source/api            MemorySystem: the contract boundary
backend/source/service        memory (ingestion, extraction, reconciliation, retrieval, forgetting, /act, guards),
                              ai (model adapter, tool loop, local embeddings), projects (product workspace)
backend/source/dal            SQLite schema, repositories (memory, search, chat, projects, memory view)
backend/source/schemas, utils validation gate; text, time, config, metrics
frontend/                     product UI (plain HTML/JS/CSS) and its local server
tests/                        213 tests (fake model, real SQLite, real local HTTP servers)
build.sh                      best-effort local model download at image build time
BUILD_STATE.md                running build log: decisions, results, limits
```

## Privacy and safety, in one place
- Text from other people or inside pasted/forwarded blocks is stored as a fact *about that content*, never as the user's, and is never followed as an instruction (enforced in code).
- Forget removes the facts, their versions and dependents, the raw source text, search entries, vectors and (in the product UI) saved chat content, then verifies and writes a counts-only receipt. It cannot reach the model provider's logs or answers already returned.
- The answer details report, per search run, the lexical and semantic hit counts and the size of the RRF ranking, mark Rank fusion as "not applied" when only one method returned results, and label each candidate with the method that found it.
- `/act` checks every write in code (quiet hours, clashes, private/forgotten/outdated content, unknown recipients, cancel, repeats, cap) before the world sees it.
- No secret is stored in the repo: `.env` is git-ignored and docker-ignored; the key is only read from the environment.

## What it uses
**Written here:** everything under `backend/`, `frontend/`, `tests/`, `build.sh`. **Python 3.12 standard library:** `sqlite3` (FTS5), `http.server`, `urllib`, `json`, `hashlib`, `re`, `threading`, `contextvars`. **Optional, installed by `build.sh` only:** [fastembed](https://github.com/qdrant/fastembed) (ONNX Runtime, tokenizers, NumPy, huggingface-hub) and the embedding model `BAAI/bge-small-en-v1.5` (local; fetched from Hugging Face at image build time only). **Runtime language model:** `deepseek/deepseek-v4.1-flash` through the event gateway. **APIs called at runtime:** the gateway's Chat Completions (with tools for `/act`) and the mock world's HTTP tools. **No** agent-memory product or framework, no vector database, no other provider. `requirements.txt` stays empty on purpose. Tests use Node only for a JavaScript syntax check.

## Honest limits
Scores so far (persona A, the offline keyword judge, which is crude): 83.0, 81.1, 81.2, 85.6, 90.7 and 87.5 in runs 1 to 6; the hidden-test score is unknown. Not verified: the arm64 build, a screen reader, browsers other than Edge. Known weak spots are listed in [MEMORY.md](MEMORY.md) section 9 and in `BUILD_STATE.md`.

## AI assistance
Coded with Claude Code (Claude Sonnet 5.5) during the event; session exports are in `prompt_history/`. The memory design decisions and their reasons are described in MEMORY.md section 8.
