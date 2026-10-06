# Production image for the reliance-under-disruption study app.
# Build:  docker build -t reliance-study:local .
# Run:    docker run --rm -p 8089:8080 -e APP_SECRET_KEY=test reliance-study:local
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=8080

# LightGBM's wheel links against OpenMP (libgomp.so.1), which slim lacks.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first so code edits don't invalidate this layer.
COPY requirements.txt .
RUN pip install -r requirements.txt

# Only what the app reads at runtime (see deploy/README.md, "What is in the image").
COPY src/ ./src/
COPY models/ ./models/
COPY data/processed/fitted_params.json data/processed/m5_weekly.npy ./data/processed/

# Unprivileged runtime user. data/ is writable only so a local run without
# DATABASE_URL can create its SQLite file; production writes to Cloud SQL.
RUN useradd --system --uid 10001 --home-dir /app --shell /usr/sbin/nologin app \
 && rm -rf src/tests \
 && find src -name '__pycache__' -prune -exec rm -rf {} + \
 && chown -R app:app /app/data
USER app

EXPOSE 8080

# Sessions live in process memory: exactly ONE worker process, many threads.
# --timeout 0 disables gunicorn's worker kill; Cloud Run enforces its own timeout.
CMD ["sh", "-c", "exec gunicorn --chdir src --workers 1 --threads 16 --timeout 0 --no-control-socket --bind :${PORT:-8080} --access-logfile - --error-logfile - interface.app:app"]
