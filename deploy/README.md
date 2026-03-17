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
Polls GitHub every 5 minutes for `claude-code-work` labelled issues.
Creates a branch, runs ClaudeCodeAgent against the issue, pushes a PR, notifies Discord.

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

| Command | Description |
|---|---|
| `!help` | Full command reference |
| `!status` | Engine, model, Gemini/GitHub config, host stats |
| `!tools` | List all registered agent tools |
| `!mcp` | MCP server status |
| `!skills` | List available Jarvis skills |
| `!weather [city]` | Current conditions + 3-day forecast |
| `!arxiv <query>` | Search recent arXiv papers |
| `!research <topic>` | Multi-source synthesis, stored to memory |
| `!summarize <url>` | Fetch and summarise any web page |
| `!memory <query>` | Search your personal memory store |
| `!digest now` | On-demand HN AI digest |
| `!sysinfo` | Host CPU/RAM/GPU stats |
| `!sysinfo full` | Detailed per-core + network stats |
| `!propose <idea>` | Draft + submit a feature proposal to GitHub |

Any message in `#jarvis`, `@Jarvis <query>`, or `!ask <query>` triggers a full agent response.

**Reaction approval:** When Jarvis posts a proposal with a GitHub issue URL, react with 👍 to
add the `claude-code-work` label (queues it for Claude Code), or 👎 to close the issue.

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
| `deep-research` | 3-search synthesis with memory recall and structured report |
| `morning-brief` | Weather + top AI story + daily insight |
| `topic-research` | Research topic and store findings to memory |
| `web-summarize` | Summarise a web page |
| `daily-digest` | Summarise recent activity |
| `code-lint` | Lint and review code |
| `data-analyze` | Analyse and interpret data |

To add a skill: create `src/openjarvis/skills/data/my-skill.toml`.
Skills can chain any registered tool via `tool_name` + `arguments_template` steps.

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
| `MAX_ISSUE_AGE_DAYS` | | `14` | Skip issues older than this |
| `CLAUDE_WORK_MODE` | | `pr` | `pr` = create branch+PR, `comment` = comment only |
| `OBSERVER_INTERVAL` | | `1800` | Observer health check interval (seconds) |
| `OBSERVER_ARXIV_HOUR` | | `8` | UTC hour for daily arXiv digest (-1 to disable) |
| `DIGEST_INTERVAL_DAYS` | | `3` | HN digest frequency |
| `DIGEST_TOP_N` | | `15` | Stories per digest |
| `DIGEST_WINDOW_HOURS` | | `72` | Story age cutoff |

---

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
