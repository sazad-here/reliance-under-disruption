# Deployment

The study app runs as a single Docker container on **Render**, with **Neon** Postgres as the study
database. Render reads [`render.yaml`](../render.yaml) from the repository root and builds the image
from the [`Dockerfile`](../Dockerfile).

## How it is set up

| Piece | Setting |
|---|---|
| Service | Render web service `reliance-study`, Docker runtime, free plan |
| Database | Neon Postgres (free tier, no expiry), reached through its pooled connection string |
| Health check | `/healthz`: answers 200 only when the database is reachable and the LightGBM model has loaded |
| Process model | exactly **one** gunicorn worker with 16 threads, because sessions live in process memory |
| Redeploys | `autoDeploy: false`: a deploy is always started by hand, so no push can interrupt a participant mid-session |

## Environment variables

| Variable | Purpose |
|---|---|
| `DATABASE_URL` | Neon pooled connection string, ending in `?sslmode=require`; entered in the Render dashboard, never committed |
| `APP_SECRET_KEY` | signs session cookies; generated once by Render and kept stable across deploys |
| `PRODUCTION=1` | secure cookies; the app refuses to start without `APP_SECRET_KEY` |
| `DESKTOP_ONLY=1` | server-side desktop gate (the task is designed for a desktop screen) |
| `DB_POOL_MAX=5` | connection-pool size, well inside the free tier's limit |
| `WAREHOUSE_3D` | `1` (default) shows the 3D warehouse; `0` shows the plain panel |

Without `DATABASE_URL` the app uses a local SQLite file, which is how it runs on a laptop.

## Deploying

1. Create the Neon database and copy its **pooled** connection string.
2. In Render: **New → Blueprint**, choose this repository, and paste the connection string as
   `DATABASE_URL` when prompted.
3. Wait for `/healthz` to report healthy, then open the site and run one session end to end.
4. Later updates: push to `master`, then press **Manual Deploy** in Render at a time when no
   participant is mid-session.

## What is in the image

Only what the app reads at runtime: `src/` (without the tests), `models/`, and
`data/processed/fitted_params.json` plus `data/processed/m5_weekly.npy`. The raw datasets are not
included.

## Running the image locally

```bash
docker build -t reliance-study:local .
docker run --rm -p 8089:8080 -e APP_SECRET_KEY=test reliance-study:local
# open http://127.0.0.1:8089
```

## Checking data integrity

```bash
python src/interface/verify_log.py --db <database> --participant <ID>
```

`verify_log.py` replays a stored session and recomputes every displayed risk estimate from the
logged state, so what a participant saw can be audited from the database.
