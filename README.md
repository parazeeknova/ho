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
