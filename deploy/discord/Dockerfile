# Jarvis Discord bot — runs the full OpenJarvis SDK in-process
# Uses local qwen3:8b via Ollama (host.docker.internal:11434)

FROM python:3.12-slim

WORKDIR /app

# Install openjarvis from source + discord.py
COPY pyproject.toml README.md ./
COPY src/ src/

RUN pip install --no-cache-dir uv && \
    uv pip install --system ".[server,tools-search]" && \
    uv pip install --system "discord.py>=2.3" pynvml croniter

COPY deploy/discord/bot.py /bot.py

CMD ["python", "/bot.py"]
