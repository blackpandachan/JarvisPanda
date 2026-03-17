# Claude Code Issue Worker — Python 3.12 + Node.js 22 (via NodeSource)

FROM python:3.12-slim

# Install Node.js 22 LTS from NodeSource (proper install, not binary copy)
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl ca-certificates && \
    curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && \
    apt-get install -y --no-install-recommends nodejs && \
    rm -rf /var/lib/apt/lists/* && \
    node --version && npm --version

# Install OpenJarvis from source
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src/ src/

RUN pip install --no-cache-dir uv && \
    uv pip install --system "."

# Copy worker script
COPY deploy/claude-worker/worker.py /worker.py

CMD ["python", "/worker.py"]
