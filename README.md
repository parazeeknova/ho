# ⚡ HO — Autonomous Job Discovery & Browser Autofill Pipeline

> *"Thy Job shall be mine"*

**HO** is an autonomous monorepo system designed for real-time job discovery, AI candidate ranking, automated browser form filling, and market intelligence. It combines a high-throughput Python ingest engine (radar, vector search, ML ranking) with a TypeScript/Stagehand browser automation worker and a Discord agent interface.

---

## 🚀 Quickstart

### Prerequisites
- **Bun** (v1.3+)
- **uv** (Python 3.14 package runner)
- **Podman** or **Docker** (for PostgreSQL + pgvector, Neo4j, Redis)
- **Node.js** (v18+)

### 1. Installation & Environment Setup
Clone the repository and set up environment variables:

```bash
cp .env.example .env
```

Ensure your `.env` contains your LLM provider credentials, Discord bot token, and channel IDs:
```env
DISCORD_BOT_TOKEN=your_bot_token
DISCORD_CHANNEL_ID=your_channel_id
OPENAI_API_KEY=your_openai_key  # or GENERALCOMPUTE_API_KEY
```

### 2. Initialize Memory & Resume Indexing
Prepare RAG vector memory and embed your resume:
```bash
bun run initm
```

---

## 📖 Commands Reference

All workspace operations are run via `bun run <command>`:

| Command | Description |
| :--- | :--- |
| `bun run run` | **Main Production Engine** — Starts infrastructure (Postgres, Redis, Neo4j), radar discovery, vector embedding, and Stagehand autofill workers. |
| `bun run status` | **Live Status Dashboard** — Real-time monitoring of job rates (obs/min, cand/min, sub/min), queue counts, active worker procs, and ML learning epoch state. |
| `bun run export` | **Export Candidates** — Dumps accepted job matches to CSV (`packages/ingest/intel/accepted_jobs.csv`). |
| `bun run initm` | **Initialize RAG Memory** — Indexes your master resume (`resume.pdf` / `RESUME_URL`) into vector memory. |
| `bun run intel` | **Market Intelligence** — Runs competitive hiring radar, salary statistics, and company research. |
| `bun run health` | **Health Diagnostic** — Checks database connections, container status, and network proxies. |
| `bun run backup` | **System Backup** — Creates gzipped volume backups of PostgreSQL and vector indexes. |
| `bun run check` | **Full Quality Check** — Runs Ruff formatting, Ruff linting, MyPy type checking, and test suites. |
| `bun run test` | **Run Test Suite** — Executes pytest and Node test runners. |

---

## ⚙️ Detailed Command Flags & Options

### 1. `bun run run` (Main Execution Engine)
Runs the full pipeline supervisor (`run_all.py`). Accepts the following flags:

```bash
bun run run [FLAGS]
```

* **`--radar-workers <N>`**: Number of parallel candidate discovery and scoring worker processes (default: `32`).
  ```bash
  bun run run --radar-workers 48
  ```
* **`--bridge-interval <seconds>`**: Interval in seconds between candidate queue drain cycles (default: `10`).
  ```bash
  bun run run --bridge-interval 5
  ```
* **`--bridge-batch <N>`**: Maximum candidates processed per drain batch (default: `20`).
  ```bash
  bun run run --bridge-batch 50
  ```
* **`--max-minutes <N>`**: Hard stop timer after N minutes of continuous operation.
  ```bash
  bun run run --max-minutes 120
  ```
* **`--no-fill`**: Runs job discovery, dorking, and ranking, but skips the autofill browser worker.
  ```bash
  bun run run --no-fill
  ```
* **`--dry-run`**: Starts infrastructure services (Postgres, Redis) to verify health without launching sweeps.
  ```bash
  bun run run --dry-run
  ```

---

### 2. `bun run status` (Live Monitoring Dashboard)
Monitors pipeline velocity, live rate/min, learning epochs, and autofill queue status:

```bash
bun run status [FLAGS]
```

* **`--watch` / `-w`**: *(Default)* Continuously updates the TUI dashboard in real time.
* **`--once`**: Prints a single static snapshot table and exits immediately.
  ```bash
  bun run status --once
  ```

---

### 3. `bun run export` (Candidate CSV Exporter)
Exports scored candidate jobs from PostgreSQL:

```bash
bun run export [FLAGS]
```

* **`--eligibility <status>`**: Filter candidates by eligibility status. Options: `accepted` (default), `near_miss`, `rejected`, `all`.
  ```bash
  bun run export -- --eligibility near_miss
  ```
* **`--mode <format>`**: Output schema format. Options: `jobs` (default CSV), `outreach` (founder socials & funding info), `all` (full JSON dump).
  ```bash
  bun run export -- --mode outreach
  ```
* **`--out <path>`**: Specify a custom CSV output file path (default: `packages/ingest/intel/accepted_jobs.csv`).
  ```bash
  bun run export -- --out ~/Desktop/my_jobs.csv
  ```

---

### 4. `python -m autofill.src.filling.resume` (Resume Deferred Job)
When an autofill job requires manual input (e.g., custom questions or OTP verification), it is deferred to Discord. Resume filing with:

```bash
python -m autofill.src.filling.resume <job_id>
```

---

## 🤖 Discord Agent Commands

The integrated Discord bot (`DiscordAgent`) allows full remote control over the pipeline via slash commands or text triggers:

* **`/analytics` or `!analytics`**: Generates a market intelligence report (Pipeline Velocity, Top Companies, Sector Signals, Skill Arbitrage) inside a dedicated thread.
* **`/status` or `!status`**: Displays active queue status, fill counts, and worker health.
* **`/memory` or `!memory`**: Shows currently loaded candidate persona context and indexed resume chunks.
* **`/health` or `!health`**: Runs diagnostic checks on databases and proxy relays.
* **`/stop` or `!stop`**: Gracefully stops active discovery and browser workers.

---

## 🏗️ Project Architecture

```text
ho/
├── packages/
│   ├── ingest/       # Python radar discovery engine, SearXNG dorker, vector store
│   ├── autofill/     # Python form solver, ScreenerRAG, Discord question bridge
│   ├── ml/           # Ranker model, LightGBM features, active learning system
│   └── node/         # Stagehand / Playwright TypeScript browser runner
├── logs/             # Centralized runtime log files
└── package.json      # Workspace root script runner
```
