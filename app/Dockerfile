# Build from the repository root so `src/` is in the context:
#     docker build -f app/Dockerfile -t forecast-api .
#     docker run --rm -p 8080:8080 forecast-api
FROM python:3.11-slim

# LightGBM links against the OpenMP runtime, which python:*-slim does not ship.
# Without it the image builds fine and the container dies at startup with
# "libgomp.so.1: cannot open shared object file".
RUN apt-get update \
    && apt-get install --no-install-recommends -y libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Dependencies next: this layer is cached across code and artifact changes, which
# is most of the rebuild time.
WORKDIR /srv
COPY app/requirements-serve.txt ./
RUN pip install --no-cache-dir -r requirements-serve.txt

# `src/` ships too -- the service builds features with the same code the backtest
# used, rather than a re-implementation that could drift.
COPY src/ ./src/
COPY app/main.py ./app/
COPY app/artifacts/ ./app/artifacts/

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8080

EXPOSE 8080

# Cloud Run injects $PORT, so the command needs a shell to expand it; `exec` then
# replaces that shell with uvicorn, keeping it PID 1 so SIGTERM shuts the container
# down cleanly instead of being swallowed.
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --workers 1"]
