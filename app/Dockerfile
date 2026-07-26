# Build from the repository root so `src/` is in the context:
#     docker build -f app/Dockerfile -t forecast-api .
#     docker run --rm -p 8080:8080 forecast-api
FROM python:3.11-slim

# Dependencies first: this layer is cached across code and artifact changes, which
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

# Cloud Run injects $PORT; `exec` keeps uvicorn as PID 1 so SIGTERM shuts it down
# cleanly instead of being swallowed by a shell.
CMD exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --workers 1
