# JarvisPanda — Personal AI Stack

A self-hosted, self-improving AI assistant built on top of [OpenJarvis](https://github.com/open-jarvis/OpenJarvis).
Local models are the primary workhorse.  Gemini Flash and Claude Code are reserved for escalation.
Discord is the primary interface.  GitHub issues are the task queue.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Discord Server                              │
│                                                                     │
│  User messages → jarvis-discord (bot)                               │
│  Proposals     → 👍/👎 reactions → GitHub issues                    │
│  Reports       → jarvis-observer, jarvis-digest (webhooks)          │
└───────────────────────┬─────────────────────────────────────────────┘
                        │
        ┌───────────────▼──────────────────────┐
        │         Three-Tier Escalation         │
        │                                       │
        │  Tier 1: qwen3:8b (Ollama on host)   │
        │     ↓ stuck after 4 turns             │
        │  Tier 2: Gemini Flash API             │
        │     ↓ needs code changes              │
        │  Tier 3: GitHub Issue → ClaudeCode    │
        └──────────────────────────────────────┘
                        │
        ┌───────────────▼──────────────────────┐
        │            Docker Stack               │
        │                                       │
        │  jarvis-server      :8000 (API)       │
        │  jarvis-discord     (bot)             │
        │  jarvis-digest      (HN cron)         │
        │  jarvis-claude-worker (issue worker)  │
        │  jarvis-observer    (health watchdog) │
        └──────────────────────────────────────┘
                        │
        ┌───────────────▼──────────────────────┐
        │          Shared Volumes               │
        │                                       │
        │  jarvis-data/  memory.db              │
        │                telemetry.db           │
        │                traces.db              │
        │  jarvis-reports/  digest markdown     │
        └──────────────────────────────────────┘
```

---

## Services

### `jarvis-discord` — Discord bot
The main interface.  Runs the full OpenJarvis orchestrator in-process (not a proxy).
- Local qwen3:8b via Ollama handles all queries
- gemini_escalate tool available when the agent needs a second opinion
- Per-user memory: each Discord user's memories are namespaced by their Discord ID
- Thread conversations carry context across messages
- Reaction approval: 👍/👎 on proposal messages manage GitHub issues

### `jarvis-server` — HTTP API
OpenAI-compatible REST API at `http://localhost:8000/v1/`.
Shares the same SQLite memory/telemetry/trace volumes.

### `jarvis-digest` — HN AI Digest
Scheduled (every 3 days by default).  Fetches top AI/ML stories from Hacker News
ranked by comment count, posts a single Discord embed.

### `jarvis-claude-worker` — Autonomous issue worker
Polls GitHub every 5 minutes for `claude-code-work` labelled issues (or wakes instantly via GitHub webhook).
Creates a branch, runs ClaudeCodeAgent against the issue, pushes a PR, notifies Discord.

**Instant pickup via webhook (recommended):** Set `GITHUB_WEBHOOK_SECRET` to any random string, expose port `9000`, then add a webhook in your repo:
`Settings → Webhooks → Add webhook` — Payload URL: `http://<host>:9000`, Content type: `application/json`, Secret: your `GITHUB_WEBHOOK_SECRET`, Events: **Issues**.
The worker wakes immediately when a label is added rather than waiting for the next poll cycle.

### `jarvis-observer` — Health watchdog
Runs every 30 minutes.  Reads trace/telemetry SQLite DBs, checks GitHub,
posts a colour-coded health report to Discord.
Opens GitHub issues automatically when a tool fails 3+ times in 24 hours.
Posts a daily arXiv AI paper digest at 8am UTC.

---

## Escalation Chain

```
User query (any complexity)
       │
       ▼
 LOCAL — qwen3:8b via Ollama
 Tools: think, web_search, url_fetch, arxiv_search,
        weather, system_monitor, retrieval, calculator,
        file_read, github_issue, gemini_escalate
       │
       │ stuck after 4 turns (coding/technical task)
       ▼
 GEMINI — gemini-2.5-flash-preview
 Single-shot: pass full context, everything tried, specific blocker
       │
       │ if Gemini resolves it → return answer to user
       │ if Gemini says "needs code changes" →
       ▼
 CLAUDE CODE — GitHub issue → claude-worker
 Issue body contains Gemini's implementation plan
 Worker creates branch, implements, opens PR
 Discord notified when PR is ready for review
```

This design keeps token costs near zero for 90%+ of interactions while
ensuring complex tasks eventually get resolved by a capable agent.

---

## Discord Commands

### Information & Status

| Command | Description |
|---|---|
| `!help` | Full command reference (auto-generated from registered commands) |
| `!status` | Local model, Gemini/GitHub config, host CPU/RAM/GPU |
| `!health` | On-demand health check: system + GitHub queue |
| `!sysinfo` | Host CPU/RAM/GPU summary |
| `!sysinfo full` | Detailed per-core + network stats |
| `!tools` | List all registered agent tools (updates automatically) |
| `!mcp` | MCP server status |
| `!skills` | List available Jarvis skill pipelines |

### Content & Research

| Command | Description |
|---|---|
| `!brief` | Morning brief: weather + top AI story + daily insight |
| `!weather [city]` | Current conditions + 3-day forecast |
| `!research <topic>` | Multi-source synthesis stored to your memory |
| `!summarize <url>` | Fetch and summarise any web page |
| `!arxiv <query>` | Search recent arXiv papers |
| `!papers [topic]` | Latest AI/ML arXiv papers (default: AI/ML) |
| `!memory <query>` | Search your personal memory store |
| `!digest now` | On-demand HN AI digest |

### GitHub Workflow

| Command | Description |
|---|---|
| `!queue` | Show issue worker queue (all label stages) |
| `!prs` | List open pull requests with merge commands |
| `!pr <number>` | PR details: files changed, description, merge command |
| `!merge <pr#>` | Squash-merge a PR and delete the source branch |

### Agent Workflows

| Command | Description |
|---|---|
| `!agent <task>` | Run a one-shot local agent task |
| `!propose <idea>` | Draft, consult Gemini, and submit a feature proposal |
| `!build-tool <desc>` | Design, spec, and propose a new Jarvis tool |
| `!run-skill <name> [args]` | Execute a registered Jarvis skill |
| `!ask-gemini <question>` | Ask Gemini Flash directly, bypass local model |

### Scheduled Tasks

| Command | Description |
|---|---|
| `!tasks` | List all tasks with status and next-run times |
| `!schedule cron 0 9 * * * <prompt>` | Create a cron-scheduled background task |
| `!schedule interval 3600 <prompt>` | Create an interval-based background task |
| `!pause-task <id>` | Pause a task (temporary; resume recomputes next_run) |
| `!resume-task <id>` | Resume a paused task |
| `!cancel-task <id>` | Permanently cancel a task |
| `!task-log <id>` | Show last 5 run results for a task |

Two tasks are seeded at startup: a daily morning brief (09:00 UTC) and a weekly AI digest (Monday 10:00 UTC). Results are posted to the `DISCORD_DIGEST_WEBHOOK` automatically via the EventBus `scheduler_task_end` event, and stored to memory for `!memory morning_brief` / `!memory weekly_digest`.

Any message in `#jarvis`, `@Jarvis <query>`, or `!ask <query>` triggers a full agent response.

**Reaction approval:** When Jarvis posts a GitHub issue URL, react 👍 to queue it for
Claude Code, or 👎 to close the issue.

### Adding New Commands

The bot uses a `CommandRegistry` with decorator-based registration:
```python
# In deploy/discord/bot.py — no changes needed to the event loop:
@cmd.exact("!mycommand", "description shown in !help")
async def _(message: discord.Message) -> None:
    await message.channel.send("Hello!")

@cmd.prefix("!mytool ", "!mytool <arg>", "does something with arg")
async def _(message: discord.Message, arg: str) -> None:
    reply = await loop.run_in_executor(_executor, do_something, arg)
    await send_chunked(message.channel, reply)
```

---

## Tool Catalog

| Tool | Category | API Key? | Description |
|---|---|---|---|
| `think` | reasoning | — | Internal chain-of-thought |
| `calculator` | utility | — | Safe math evaluation |
| `retrieval` | memory | — | Search OpenJarvis memory store |
| `web_search` | web | Tavily (optional) | Web search |
| `url_fetch` | web | — | Fetch + clean any web page |
| `arxiv_search` | research | — | Search arXiv papers |
| `weather` | utility | — | Open-Meteo weather + forecast |
| `system_monitor` | utility | — | CPU/RAM/disk/GPU via psutil |
| `file_read` | file | — | Read files in workspace |
| `gemini_escalate` | escalation | GEMINI_API_KEY | Second-opinion via Gemini Flash |
| `github_issue` | escalation | GITHUB_TOKEN | Create escalation issues |

To add a new tool:
1. Create `src/openjarvis/tools/my_tool.py` implementing `BaseTool`
2. Decorate with `@ToolRegistry.register("my_tool")`
3. Add `import openjarvis.tools.my_tool` to `src/openjarvis/tools/__init__.py`
4. It appears in `!tools` and is available to the agent automatically

---

## Agent Reference

All agents implement `BaseAgent.run()`. The `jarvis-discord` container uses `orchestrator` for all queries.

| Agent key | Best for |
|---|---|
| `orchestrator` | General multi-step tool use (default for JarvisPanda) |
| `simple` | Single-turn Q&A, no tools |
| `native_react` | When you need visible Thought-Action-Observation traces |
| `native_openhands` | Inline Python code generation and execution |
| `rlm` | Document-scale contexts (stores variables in REPL) |
| `claude_code` | Spawns Claude Code Node.js subprocess (worker only, requires Node 22+) |
| `sandboxed` | Wraps any agent in Docker/Podman with disabled networking |

Switch agent per-query: `j.ask(prompt, agent="native_react", tools=[...])`.
Switch the server default: `[server] agent = "native_react"` in `config.persistent.toml`.

---

## Memory Backends

The stack currently uses **SQLite/FTS5** (zero deps, BM25 ranking, disk-persistent). Upgrade by changing `[tools.storage] default_backend`:

| Backend | Config key | Extra deps | Quality | Persistent |
|---|---|---|---|---|
| SQLite/FTS5 | `sqlite` | none | good | ✓ |
| FAISS | `faiss` | `faiss-cpu`, `sentence-transformers` | high | ✗ (memory) |
| ColBERTv2 | `colbert` | `colbert-ai`, `torch` | highest | ✗ (memory) |
| BM25 | `bm25` | `rank_bm25` | good | ✗ (memory) |
| Hybrid (RRF) | `hybrid` | depends on sub-backends | highest | depends |

To enable FAISS for better semantic retrieval:
```toml
[tools.storage]
default_backend = "faiss"
db_path = "/data/memory.db"
```
Add `faiss-cpu sentence-transformers` to the discord/server Dockerfiles.

**Chunking defaults:** 512 tokens, 64 overlap, splits on `\n\n` paragraph boundaries.
**Context injection:** top_k=5, min_score=0.1, max_context_tokens=1500 (set in `[memory]`).

---

## MCP Support

OpenJarvis supports the [Model Context Protocol](https://modelcontextprotocol.io/).
Add servers to `deploy/docker/config.persistent.toml`:

```toml
[tools.mcp]
enabled = true

[[tools.mcp.servers]]
name    = "filesystem"
command = ["npx", "-y", "@modelcontextprotocol/server-filesystem", "/data"]

[[tools.mcp.servers]]
name    = "github"
command = ["npx", "-y", "@modelcontextprotocol/server-github"]
env     = { GITHUB_PERSONAL_ACCESS_TOKEN = "${GITHUB_TOKEN}" }
```

All MCP tools are discovered automatically and appear in `!tools`.

---

## Skills

Skills are TOML-defined multi-step pipelines in `src/openjarvis/skills/data/`.

| Skill | Description |
|---|---|
| `deep-research` | Multi-turn: web_search × 3 + arxiv + memory recall → structured report |
| `morning-brief` | Weather + top AI story + daily insight → stored to memory |
| `topic-research` | Research topic and store findings to memory |
| `web-summarize` | Fetch + summarise a web page |
| `daily-digest` | Summarise recent activity |
| `code-lint` | Lint and review code |
| `data-analyze` | Analyse and interpret data |
| `knowledge-extract` | Extract structured knowledge from documents |
| `pdf-summarize` | Summarise a PDF document |
| `security-scan` | Security review of code or config |

The `deep-research` skill mirrors the multi-turn research pattern from the [OpenJarvis docs](https://open-jarvis.github.io/OpenJarvis/tutorials/deep-research/): OrchestratorAgent loops through `web_search → think → memory_store → memory_search → synthesise` until `max_turns` or completion.

**Skill TOML format:**
```toml
[skill]
name        = "my-skill"
version     = "0.1.0"
description = "What this skill does"
author      = "jarvis"

[[skill.steps]]
tool_name          = "web_search"
arguments_template = '{"query": "{topic}"}'
output_key         = "search_results"

[[skill.steps]]
tool_name          = "think"
arguments_template = '{"thought": "Summarise: {search_results}"}'
output_key         = "summary"
```

Placeholders `{key}` are resolved from execution context; each step's `output_key` becomes available to subsequent steps.

---

## Setup

### Prerequisites

- Docker Desktop (Windows/Mac) or Docker Engine (Linux)
- [Ollama](https://ollama.ai) installed natively (not in Docker)
- qwen3:8b pulled: `ollama pull qwen3:8b`
- A Discord bot token and webhook URL
- Optional: Gemini API key, Anthropic API key, Tavily API key, GitHub token

### 1. Clone and configure

```bash
git clone https://github.com/blackpandachan/JarvisPanda.git
cd JarvisPanda
cp deploy/.env.example deploy/.env
# Edit deploy/.env with your values
```

### 2. Discord bot setup

1. Go to [discord.com/developers/applications](https://discord.com/developers/applications)
2. Create an application → Bot → copy the token → `DISCORD_BOT_TOKEN`
3. Enable **Message Content Intent** and **Server Members Intent**
4. Invite with permissions: `Send Messages`, `Create Public Threads`, `Read Message History`,
   `Add Reactions`, `Manage Messages`
5. Create a `#jarvis` channel (or set `DISCORD_CHANNEL_NAME` to match your channel)
6. Create a webhook in the channel → `DISCORD_DIGEST_WEBHOOK`

### 3. Start the stack

```bash
cd deploy/docker
docker compose -f docker-compose.persistent.yml up -d --build
docker compose -f docker-compose.persistent.yml logs -f
```

### 4. Verify

```
# In Discord #jarvis:
!status       → should show model + Gemini/GitHub config
!weather NYC  → should return weather data
!sysinfo      → should show host CPU/RAM
```

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `DISCORD_BOT_TOKEN` | ✓ | — | Discord bot token |
| `DISCORD_CHANNEL_NAME` | | `jarvis` | Channel Jarvis listens to |
| `DISCORD_PREFIX` | | `!ask` | Command prefix |
| `DISCORD_DIGEST_WEBHOOK` | | — | Webhook for digest/observer posts |
| `GITHUB_TOKEN` | | — | PAT with repo scope |
| `GITHUB_REPO` | | — | `owner/repo` |
| `ANTHROPIC_API_KEY` | | — | For ClaudeCodeAgent worker |
| `GEMINI_API_KEY` | | — | For Gemini escalation tier |
| `GEMINI_MODEL` | | `gemini-2.5-flash-preview` | Gemini model ID |
| `TAVILY_API_KEY` | | — | Richer web search (falls back to DuckDuckGo) |
| `DEFAULT_WEATHER_CITY` | | — | Default city for `!weather` |
| `AGENT_EXTRA_TOOLS` | | — | Comma-separated extra tool names to add |
| `POLL_INTERVAL` | | `300` | Issue worker poll interval (seconds) |
| `GITHUB_WEBHOOK_SECRET` | | — | Shared secret for instant issue pickup via webhook (optional) |
| `GITHUB_WEBHOOK_PORT` | | `9000` | Port the webhook receiver listens on |
| `MAX_ISSUE_AGE_DAYS` | | `14` | Skip issues older than this |
| `MERGE_STRATEGY` | | `manual` | `manual` = wait for `!merge`/👍 · `auto` = merge after delay · `comment` = legacy |
| `MERGE_DELAY_MINUTES` | | `15` | Minutes before auto-merge (only when `MERGE_STRATEGY=auto`) |
| `DEFAULT_BRANCH` | | `main` | Base branch for PRs |
| `OBSERVER_INTERVAL` | | `1800` | Observer health check interval (seconds) |
| `OBSERVER_ARXIV_HOUR` | | `8` | UTC hour for daily arXiv digest (-1 to disable) |
| `DIGEST_INTERVAL_DAYS` | | `3` | HN digest frequency |
| `DIGEST_TOP_N` | | `15` | Stories per digest |
| `DIGEST_MIN_SCORE` | | `30` | Minimum HN score to include |
| `SCHEDULER_DB_PATH` | | `/data/scheduler.db` | Path for task scheduler SQLite DB |
| `AGENT_EXTRA_TOOLS` | | — | Comma-separated extra tool names beyond auto-discovered set |

---

## PR Lifecycle & Merge Workflow

When an issue labelled `claude-code-work` is created, the claude-worker:

```
1. Claims the issue (claude-code-in-progress label)
2. Creates branch: claude-code/issue-N-slug from main
3. Runs ClaudeCodeAgent (up to 30 min)
4. Pushes branch, opens PR linked to the issue
5. Posts an "Implementation Guide" on BOTH the issue and PR:
     - Files changed with +/- counts
     - Commit list
     - Verification steps (git checkout + pytest)
     - Merge command: !merge <pr#>
6. Notifies Discord with PR link and instructions
7. Labels issue as claude-code-done
```

**MERGE_STRATEGY options:**

| Strategy | Behaviour |
|---|---|
| `manual` (default) | Post PR + guide, wait for `!merge <pr#>` or 👍 reaction |
| `auto` | Post PR + guide, auto-merge after `MERGE_DELAY_MINUTES` (default 15) if still open |
| `comment` | Legacy: post agent output as issue comment, no branch/PR |

**Merge via Discord:**
```
!prs              → see open PRs
!pr 42            → see what changed in PR #42
!merge 42         → squash-merge PR #42 + delete branch
```

**Merge via local agent:**
The agent can call `github_merge(pr_number=42)` directly.
If merge fails (conflicts), the agent escalates to Gemini for a resolution plan.

## How the Self-Improvement Loop Works

```
1. Users interact via Discord
          ↓
2. Agent attempts task with local model
          ↓
3. If stuck → gemini_escalate (Tier 2)
          ↓
4. If needs code → github_issue with Gemini's plan (Tier 3)
          ↓
5. claude-worker polls GitHub, creates branch, works the issue
          ↓
6. ClaudeCodeAgent commits changes, pushes branch, opens PR
          ↓
7. Discord notified: "PR ready for review" embed
          ↓
8. Human reviews + merges (or !propose + 👍 for agent-proposed features)
          ↓
9. jarvis-observer watches for patterns:
   - Tools failing repeatedly → auto-opens fix issues
   - High turn-limit hit rate → auto-opens system prompt improvement issue
          ↓
10. The stack improves itself over time
```

---

## Adding More Messaging Channels

OpenJarvis has 26+ built-in channel integrations (Telegram, Slack, WhatsApp, Signal, Teams, Matrix, IRC, and more) registered in `ChannelRegistry`. The JarvisPanda Discord bot is a custom `discord.py` integration for maximum control — but you can add additional platforms alongside it.

```bash
# Check what's available
jarvis channel list
jarvis channel status
```

**Slack example** — add to `deploy/.env`:
```env
SLACK_BOT_TOKEN=xoxb-...
SLACK_APP_TOKEN=xapp-...
```

Then create a channel recipe `deploy/slack/messaging.toml`:
```toml
[channel]
default = "slack"

[agent]
type        = "orchestrator"
max_turns   = 5
temperature = 0.3
tools       = ["think", "web_search", "memory_store", "memory_search"]
```

Load with `SystemBuilder`:
```python
from openjarvis.recipes import load_recipe
from openjarvis import SystemBuilder

recipe = load_recipe("deploy/slack/messaging.toml")
system = SystemBuilder(**recipe.to_builder_kwargs()).build()
```

The same agent + memory + tools stack that powers Discord will be available on any additional channel. Per-user memory works identically as long as the channel provides a stable user ID.

---

## Ideas / Roadmap

These are features the agents can propose and implement themselves via `!propose`:

- **`remind` tool** — natural language reminders stored to the OpenJarvis scheduler (`!remind me about X in 3 days`)
- **`github_search` tool** — search issues/code across repos for self-referential tasks
- **Per-user preferences** — agents ask about and remember user preferences (timezone, units, topics of interest)
- **Digest personalisation** — each user's digest filtered to their stored interests
- **Slack/Telegram channels** — OpenJarvis supports these via `ChannelRegistry`
- **arXiv alerts** — observer watches for new papers on user-specified topics
- **Automated dependency audits** — weekly `pip audit` + cargo audit, reported to Discord
- **Model performance tracking** — trace DB analysis to auto-tune which tasks go to which model
- **Local code indexing** — file_read + retrieval indexed over the repo so agents can answer codebase questions directly
- **Two-phase issues** — Claude plans only (fast, cheap), local agent implements, ClaudeCode as final fallback

---

## Project Structure

```
JarvisPanda/
├── src/openjarvis/           # OpenJarvis SDK (Python package)
│   ├── tools/                # Tool implementations
│   │   ├── weather.py        # Open-Meteo weather
│   │   ├── arxiv_search.py   # arXiv paper search
│   │   ├── url_fetch.py      # Web page fetcher
│   │   ├── system_monitor.py # Host resource stats
│   │   ├── gemini_escalate.py# Gemini Flash escalation
│   │   └── github_issue.py   # Issue creation / escalation
│   └── skills/data/          # TOML skill pipelines
│       ├── deep-research.toml
│       └── morning-brief.toml
│
├── deploy/
│   ├── discord/              # Discord bot
│   │   ├── bot.py            # Main bot (CommandRegistry, agent loop)
│   │   └── Dockerfile
│   ├── digest/               # HN AI digest scheduler
│   │   ├── digest.py
│   │   └── Dockerfile
│   ├── claude-worker/        # Autonomous issue worker
│   │   ├── worker.py         # PR mode + label lifecycle
│   │   └── Dockerfile
│   ├── observer/             # Health watchdog
│   │   ├── observer.py       # Trace/telemetry analysis + auto-issues
│   │   └── Dockerfile
│   ├── docker/
│   │   ├── docker-compose.persistent.yml
│   │   ├── config.persistent.toml
│   │   └── Dockerfile.dev
│   └── .env.example          # Template — copy to .env
│
└── rust/                     # Rust workspace (PyO3 bindings)
```

---

*Built on [OpenJarvis](https://github.com/open-jarvis/OpenJarvis) · Deployed at [JarvisPanda](https://github.com/blackpandachan/JarvisPanda)*
