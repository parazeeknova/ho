# ho

One-command setup of the memory base (resume + persona RAG) for a fresh checkout.

## Setup

```sh
make init-memory
```

Checks Postgres + embedding server (auto-starts it if needed), indexes the resume into `resume_embeddings`, and builds the persona into `persona_embeddings` + `persona.txt` (interactively grills you when `persona.json` is missing). Prints a memory summary when done.

## Commands

| Command | Description |
|---|---|
| `make init-memory` | Full memory base setup: Postgres, embed server, resume, persona. |
| `make grill-persona` | Interactive wizard: 7 identity fields + 12 personal Q&A + optional extra Q&A; writes `persona.json` and rebuilds persona memory. |
| `make index-resume` | Download the resume from `RESUME_URL` (or `--path`), chunk it, and index into `resume_embeddings`. |

## Options

```sh
uv run python scripts/init_memory.py --no-resume     # skip resume indexing
uv run python scripts/init_memory.py --no-grill      # skip the wizard
uv run python scripts/init_memory.py --grill         # force re-grill
uv run python scripts/grill_persona.py --no-build    # only write persona.json
uv run python scripts/index_resume.py --path <file>  # index a local resume
```

## Autofill

When the autofill pipeline meets a screener question it cannot answer from the
persona knowledge base, it prompts the user on Telegram (`TELEGRAM_BOT_TOKEN` /
`TELEGRAM_CHAT_ID`) and waits for a reply (`AUTOFILL_QUESTION_TIMEOUT`, default
300s). The answer is applied to the form and learned into the persona KB.
Without Telegram configured, such a question fails the run loudly instead of
submitting blank answers.

### Overnight mode

When `OVERNIGHT_LOOP=true`, no human is present to answer questions or review
forms. Fully-fillable jobs are filled and **submitted automatically**
(`submitAllowed`); a job that hits an unknown question is never submitted — the
question is recorded, the job is marked `deferred`, and it is listed in the
morning digest for you to answer and resume (`python -m autofill resume`).

As a safety net, the overnight worker arms an activity watchdog
(`AUTOFILL_ACTIVITY_TIMEOUT_MS`, default 360000 = 6 min): if the browser stops
making observable progress (a stuck fill or submit), the runner aborts the job
instead of hanging until the hour-long DB lease. Any runner/status/RPC activity
resets the idle timer, so a healthy run is never cut off.

### Workday portals

Workday career sites are multi-step wizards gated behind a sign-in / account
creation screen, and most tenants (Intel, Salesforce, …) do not offer a guest
apply path. The adapter always clicks "Apply Manually" when offered. Because
Workday accounts are **per-tenant** (the same email/password won't already
exist on most companies' sites), it then **creates an account** on the tenant
using `WORKDAY_EMAIL` / `WORKDAY_PASSWORD`, and falls back to signing in with
those credentials when the account already exists. Set both variables in
`.env` so Workday applications can proceed.
