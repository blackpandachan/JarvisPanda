# Jarvis Observer Agent
# Lightweight watchdog — no GPU, no Node.js, just Python + httpx

FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir httpx

COPY deploy/observer/observer.py /observer.py

CMD ["python", "/observer.py"]
