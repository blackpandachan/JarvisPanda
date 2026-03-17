# Jarvis Claude Code Label Manager
# Ensures claude-code-* labels exist and optionally receives GitHub webhooks.
# AI implementation work is handled by GitHub Actions (claude-issue-worker.yml).

FROM python:3.12-slim

# httpx only — no Node.js, no Claude CLI needed
RUN pip install --no-cache-dir httpx

COPY deploy/claude-worker/worker.py /worker.py

CMD ["python", "/worker.py"]
