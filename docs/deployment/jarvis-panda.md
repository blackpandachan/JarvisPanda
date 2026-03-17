---
title: JarvisPanda Deployment
description: Running the full JarvisPanda autonomous AI stack on Docker
---

# JarvisPanda Stack

JarvisPanda is a personal autonomous AI deployment built on top of OpenJarvis. It runs as five Docker services and adds a Discord bot, a GitHub-integrated AI worker, a health observer, and a three-tier escalation loop.

## Prerequisites

- Docker + Docker Compose
- Ollama running locally (or accessible on the network)
- A Discord bot token
- A GitHub PAT with `repo` scope (read/write issues + PRs)
- The Claude Code GitHub App installed on your repo (`github.com/apps/claude`)
- A Discord webhook URL for notifications

## Quick Start

```bash
cd deploy/docker
cp ../. env.example ../.env     # fill in credentials
docker compose -f docker-compose.persistent.yml up -d --build
docker compose -f docker-compose.persistent.yml logs -f
```

## Services

| Service | Role |
|---|---|
| `jarvis-server` | OpenAI-compatible API on `:8000` (Ollama backend) |
| `jarvis-discord` | Discord bot running OrchestratorAgent + full command set |
| `jarvis-digest` | Hacker News AI digest posted to Discord every N days |
| `jarvis-claude-worker` | Ensures `claude-code-*` GitHub labels exist; webhook receiver |
| `jarvis-observer` | Health watchdog: auto-opens issues, posts arXiv digests |

Shared SQLite volumes at `/data/`: `memory.db`, `telemetry.db`, `traces.db`, `scheduler.db`.

## Environment Variables

Copy `deploy/.env.example` to `deploy/.env` and fill in:

| Variable | Required | Description |
|---|---|---|
| `DISCORD_BOT_TOKEN` | ✓ | Discord bot token |
| `GEMINI_API_KEY` | ✓ | Gemini Flash (Tier 2 escalation) |
| `GITHUB_TOKEN` | ✓ | PAT: repo read/write + issues + PRs |
| `GITHUB_REPO` | ✓ | `owner/repo` (e.g. `blackpandachan/JarvisPanda`) |
| `DISCORD_DIGEST_WEBHOOK` | ✓ | Webhook URL for digest/observer/PR notifications |
| `TAVILY_API_KEY` | | Richer web search (falls back to DuckDuckGo) |
| `GITHUB_WEBHOOK_SECRET` | | Optional: HMAC secret for fast issue pickup |
| `GITHUB_WEBHOOK_PORT` | | Webhook listener port (default `9000`) |

Also add `DISCORD_DIGEST_WEBHOOK` as a **GitHub Actions secret** (`Settings → Secrets → Actions → New repository secret`) so the CI workflow can notify Discord when a PR is opened.

**No `ANTHROPIC_API_KEY` required.** Tier 3 AI work uses the Claude Code GitHub App via subscription OAuth.

## Three-Tier Escalation

```
Tier 1  qwen3:8b via Ollama (local, free)
   ↓  stuck after 4 turns on a technical task
Tier 2  Gemini Flash via GEMINI_API_KEY (cloud, cheap)
   ↓  Gemini confirms code changes needed
Tier 3  GitHub issue (claude-code-work label)
         → GitHub Actions → @claude mention
         → Claude Code GitHub App implements fix
         → PR opened → Discord notified
         → local agents verify + merge
```

The Discord `!propose <task>` command cuts an issue with the `claude-code-work` label to trigger Tier 3.

## PR Lifecycle

```
!propose <task>  in Discord
  → GitHub issue created with label claude-code-work
  → .github/workflows/claude-issue-worker.yml triggers
  → @claude comment posted (Claude Code GitHub App)
  → Claude Code implements fix on branch, opens PR
  → Discord notified via DISCORD_DIGEST_WEBHOOK
  → !merge <pr#>  or  👍 on issue comment  to merge
```

## Active Config

The runtime OpenJarvis config is `deploy/docker/config.persistent.toml`. Key settings:

```toml
[engine]
default = "ollama"

[engine.ollama]
host = "http://host.docker.internal:11434"   # Ollama on the host

[intelligence]
default_model = "qwen3:8b"
temperature = 0.7
max_tokens = 2048

[agent]
default_agent = "orchestrator"
max_turns = 8
tools = "web_search,arxiv_search,think,memory_store,memory_search,gemini_escalate,github_issue,github_merge,weather"
context_from_memory = true

[tools.storage]
default_backend = "sqlite"
db_path = "/data/memory.db"
```

## Scheduler

Two tasks are seeded at startup in `jarvis-discord`:

- **Daily morning brief** — `0 9 * * *` UTC: summarises AI news, stores to memory, posts to Discord
- **Weekly AI digest** — `0 9 * * 1`: deep arXiv + HN research, posts to Discord

Manage via Discord commands: `!tasks`, `!schedule`, `!pause-task`, `!resume-task`, `!cancel-task`, `!task-log`.

## Useful Docker Commands

```bash
# Rebuild and restart a single service
docker compose -f deploy/docker/docker-compose.persistent.yml up -d --build jarvis-discord

# Tail logs for one service
docker compose -f deploy/docker/docker-compose.persistent.yml logs -f jarvis-discord

# Check health of all services
docker compose -f deploy/docker/docker-compose.persistent.yml ps

# Open a shell in the discord container
docker compose -f deploy/docker/docker-compose.persistent.yml exec jarvis-discord bash
```
