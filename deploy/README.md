# Deploying the study

This guide puts the study online so volunteers from the university's SONA
participant pool can take part from home on a desktop computer. It is written for a researcher, not a server administrator.
Follow the steps in order. Total time for a first deployment is about an hour, most
of it waiting.

## Two hosting options

This folder has two deployment scripts. **They deploy the same Docker image and the same app** --
only the cloud differs. Pick one.

| | `deploy-azure.sh` (Azure) | `deploy.sh` (Google Cloud) |
|---|---|---|
| Credit | USD 100, **12 months**, renewable while enrolled | USD 300, 90 days, once |
| **Credit card** | **Not needed** -- university email verifies you | Required at signup |
| Source | Azure for Students, via the GitHub Student Developer Pack | Google Cloud free trial |
| Services | Container Apps + PostgreSQL flexible server | Cloud Run + Cloud SQL + Firebase Hosting |
| Public URL | `https://<app>.<region>.azurecontainerapps.io` | `https://<project>.web.app` |
| Commands | `infra`, `release`, `status`, `export`, `allow-me`, `stop`, `all` | `infra`, `release`, `hosting`, `all` |

**If you have the GitHub Student Pack, use Azure** -- no credit card, and the credit lasts four times
as long. Steps 2-4 below are written for Google Cloud; the Azure equivalents are:

```bash
# once, on your computer
winget install Microsoft.AzureCLI      # or https://aka.ms/installazurecliwindows
az login                               # opens a browser; use your student account

# once, in deploy/deploy-azure.sh: set PG_SERVER to something globally unique
#   PG_SERVER="reliance-db-ab12cd"

bash deploy/deploy-azure.sh infra      # ~10 min: resource group, PostgreSQL, environment
bash deploy/deploy-azure.sh release    # ~10 min first time: builds the image in the cloud
bash deploy/deploy-azure.sh status     # prints the URL and the health check
```

`infra` writes the database password to `deploy/.azure-db-url` (gitignored, chmod 600). Keep it --
`release` and `export` both read it. `release` never rotates an existing secret.

**Before your first `export`, run `bash deploy/deploy-azure.sh allow-me`.** `infra` creates the
database with `--public-access 0.0.0.0`, which permits *Azure services only* -- not your own
computer. Without a firewall rule for your address `pg_dump` simply times out. `allow-me` looks up
this computer's current public IP and adds that rule. Home IP addresses change, so run it again
whenever `export` stops connecting.

**After collection**, `bash deploy/deploy-azure.sh export` dumps the database into `data/`, and
`stop` scales the app to zero so it stops consuming credit without losing anything.

### Which commands the freeze blocks

`deploy/FREEZE` blocks **changes only** -- `infra`, `release` and `all`. The commands you need
*during* collection keep working: `status` (is it alive?), `export` (the daily backup), `stop` and
`allow-me`. Before 5 Oct 2026 the guard sat in `preflight()` and blocked all six, which would have
made the daily export impossible exactly when it mattered.

Everything else in this README -- the verification checklist, the deploy freeze, data protection,
the rollback advice -- applies to both.

## Plan and schedule (written 4 Oct 2026)

Dates assume the registered timeline: pilot **13 Oct**, collection **19 Oct - 1 Nov**.
Deploying on the Google Cloud free trial, so the study should cost **nothing**.

### The money rules, in order of importance

1. **Never click "Upgrade", "Activate full account" or "Enable billing".** Google prompts for this
   repeatedly. The trial billing account **auto-closes** when the $300 credit or the 90 days run
   out; it does not roll over into charges. Upgrading manually is the only way to be billed.
2. **Set a budget alert at USD 100 on day one** (Billing -> Budgets & alerts). Two minutes.
3. **Sign up close to the deployment date, not now-ish-plus-months.** The clock is 90 days from
   signup. Signing up around 6 Oct covers you to early January; collection ends 1 Nov.
4. Expected spend for the whole study is roughly **USD 30-40** of the 300 (see the cost table
   below), so there is 7-10x headroom. Use `SQL_TIER="db-f1-micro"` to widen it further.
5. **Eligibility:** the trial requires that you have never been a paying Google Cloud, Firebase or
   Google Maps Platform customer, and have not used the trial before. Check this first -- the rest
   of the plan depends on it.

### Critical path

**Ethics approval is the long pole, not the deployment.** The deployment is about an hour of work
and can be done in a single afternoon. Approval cannot. If the committee has not approved this
protocol version, the 13 Oct pilot is the thing at risk, not the server. Confirm that date before
spending effort here.

### Phase 0 - decisions and paperwork (now to ~10 Oct). No spend, no server.

| Who | Task |
|---|---|
| You | Confirm ethics approval covers this protocol version (`docs/ETHICS-AMENDMENT.md`, especially the sect. 0 recruitment change) |
| You | Supervisor sign-off on OSF open questions 12-17 (including sample size and the 3D interface, Q16) |
| You | Submit the OSF pre-registration **before any data is collected** |
| You | Fill the placeholders in `src/interface/templates/info.html`: ethics committee and reference, institution, your name, supervisor, study email, the two retention periods, the date |
| You | Ask the university whether they require institutional hosting for participant data. If yes, stop here -- that route is free and may be mandatory |
| You | Create the SONA study and get its participant-ID URL parameter |
| Claude | Final check of the test suite, dry run and browser demo before anything is deployed |

### Phase 1 - deploy and verify (~10-12 Oct). Still no participants.

| Who | Task |
|---|---|
| You | Create the Google Cloud project with the free trial, enable Firebase on it, set the budget alert |
| You | Install and authenticate the gcloud CLI (Step 2 below) |
| You | Put the project ID into `deploy/deploy.sh` and `.firebaserc` (Step 3), or give it to Claude to do |
| You | Run `bash deploy/deploy.sh all` (Step 4). About an hour, most of it waiting |
| Claude | Work through the Step 5 verification checklist against the live URL: consent, screener, a full session, the debrief, and `verify_log.py` on the resulting database |
| You | Confirm the SONA link carries the participant ID through to the consent page |

### Phase 2 - pilot (13 Oct)

Roughly 10 participants. Afterwards: run the analysis scripts on the pilot data, confirm the logs
replay, and check the exclusion rules behave. Fix anything found **before** 19 Oct, because the
freeze starts then.

### Phase 3 - collection (19 Oct - 1 Nov)

| Who | Task |
|---|---|
| You | `touch deploy/FREEZE` before the first participant. The deploy script then refuses to run |
| You | Do not redeploy. One instance holds live sessions in memory; a redeploy drops them |
| Claude | Optional daily collection report (completion rate, exclusions per condition, drop-out) |
| You | Watch the budget alert |

### Phase 4 - after collection (from 1 Nov)

| Who | Task |
|---|---|
| You | `rm deploy/FREEZE` |
| You | Export the data (see "Exporting the data for analysis" below) and take a final backup |
| Claude | Run `src/analysis/run_all.py` on the real data and fill the paper's Sect. 4.4 and Table 3 |
| You | Stop paying: the two commands in the cost section below. Do not delete anything until the retention plan allows it |

### If the free trial is not available to you

In order of preference: university hosting (free, possibly mandatory); **Render** free tier (the app
needs only 161 MB, measured, and survives sleeping because sessions rehydrate from the database --
but its free Postgres expires after 30 days, so create it close to the pilot); **Oracle Cloud Always
Free** (a real VM, free indefinitely, but you install Docker, Postgres and HTTPS yourself).

**Not Vercel.** It runs stateless functions; this app is one stateful process with sessions in
memory, and its 130 tests were written against that design.

## How it fits together

```
Participant's browser
      |
      v
Firebase Hosting   https://YOUR-PROJECT.web.app      (the public address, HTTPS)
      |  forwards every path, plus one cookie named __session
      v
Cloud Run          service "reliance-study", europe-west1
      |  exactly ONE always-on instance running the Flask app (gunicorn, 1 process, 16 threads)
      v
Cloud SQL          PostgreSQL 16 "reliance-db", europe-west1, daily backups + point-in-time recovery
```

Two secrets live in Secret Manager, not in the code: `APP_SECRET_KEY` and
`DATABASE_URL`. The service also runs with `PRODUCTION=1` and `DESKTOP_ONLY=1`.
(`PROLIFIC_COMPLETION_CODE` is no longer used: participation is unpaid and
recruited through SONA, so there is no completion code. The app runs normally
with it unset.)

**Why only one instance?** Each participant's game state is held in the server's
memory. If Cloud Run started a second copy, a participant could land on a copy that
doesn't have their game. The script fixes the instance count at 1, never more or less.
One instance comfortably handles the expected load.

**What is in the image:** `src/` (without tests), `models/live_risk_lgbm.txt`,
`models/live_risk_meta.json`, `data/processed/fitted_params.json`,
`data/processed/m5_weekly.npy`. Nothing else: no raw data, no SQLite files, no
PDFs. `.dockerignore` and `.gcloudignore` enforce this.

## Files

| File | Purpose |
|---|---|
| `Dockerfile` | Builds the app image (Python 3.13, gunicorn, one process). |
| `requirements.txt` | Exact package versions for the image. `lightgbm` must stay at the version in `models/live_risk_meta.json`. |
| `requirements-dev.txt` | Adds pytest and Playwright for local testing. |
| `.dockerignore`, `.gcloudignore` | Allow-lists of what goes into the image or gets uploaded to Cloud Build. |
| `firebase.json`, `.firebaserc` | Firebase Hosting config: every path goes to Cloud Run, and `/api/*` is never cached. |
| `deploy/deploy.sh` | Every cloud command, in order. It does nothing until you run it. |
| `deploy/public/robots.txt` | Keeps the study out of search engines. |

## Fixes applied 5 October 2026 (before the first real run)

Neither cloud script had ever been executed, so these were found by reading rather than by failing:

1. **`.dockerignore` excluded the `Dockerfile` itself.** The file is an allow-list that starts with
   `*`, and `Dockerfile` was never re-included. Local `docker build` is unaffected, because Docker
   reads the Dockerfile separately, and `.gcloudignore` already worked around it with `!Dockerfile`.
   But `az containerapp up --source .` filters its upload through `.dockerignore`, so the remote
   build would not have found a Dockerfile and would have fallen back to buildpack detection --
   producing a silently wrong image. Fixed by adding `!Dockerfile`.
2. **The freeze guard blocked `status`, `export` and `stop`** (see above).
3. **The database firewall had no rule for your own computer**, so `export` could not connect. Added
   the `allow-me` command, and `export` now names this as the likely cause when `pg_dump` fails.
4. **`az containerapp` needs a CLI extension** that is not part of the base install. `preflight` now
   runs `az extension add --upgrade --name containerapp`, because the interactive install prompt
   fails under `-o none`.
5. **`PG_SERVER` was empty** and is now set to a unique name.

Still never executed against a real cloud: expect first-run snags around resource-provider
registration and quota on a student subscription.

---

## Step 1: What you must create yourself

The script can't create these for you, because they involve your identity and your money.

1. **A Google account** that will own the project. Use an institutional account if
   your university allows it.
2. **A Firebase project.** Go to <https://console.firebase.google.com>, click
   *Add project*, and give it a name (for example `reliance-study`). Google Analytics
   isn't needed. Note the **project ID** it shows, such as `reliance-study-4f2a1`.
   Every Firebase project is also a Google Cloud project with the same ID.
3. **Billing (Blaze plan).** In the Firebase console go to *Upgrade*, then *Blaze
   (pay as you go)*, and link a billing account. Cloud Run, Cloud SQL and Hosting
   rewrites to Cloud Run all require billing. Then set a budget alert in the Google
   Cloud console (*Billing*, then *Budgets & alerts*), for example EUR 60 per month
   with emails at 50 %, 90 % and 100 %.
4. **The region.** Everything the script creates is in `europe-west1` (Belgium).
   Keep it there; see *Data protection* below.
5. **The SONA study** (after deployment, once you have the live URL). Set it up
   as an online (external) study whose link carries the participant ID. The
   typical SONA pattern is `https://YOUR-PROJECT.web.app/?id=%SURVEY_CODE%`:
   SONA replaces `%SURVEY_CODE%` with each participant's identification code.
   Confirm the exact placeholder in your institution's SONA study settings, and
   that the study's start page reads the same parameter name. No completion
   code is needed.

## Step 2: Install the tools (Windows)

1. **Google Cloud CLI.** Download and run the installer from
   <https://cloud.google.com/sdk/docs/install#windows>. Accept the defaults.
2. **Node.js LTS** (if `node --version` doesn't already work) from <https://nodejs.org>,
   then the **Firebase CLI**: open a new terminal and run `npm install -g firebase-tools`.
3. **Git for Windows** (provides *Git Bash*, which the script needs). It is already
   installed if the `bash` command works.
4. **Log in.** Open *Git Bash* in the project folder and run:

   ```bash
   gcloud auth login              # opens a browser; use the account that owns the project
   gcloud config set project YOUR-PROJECT-ID
   firebase login                 # opens a browser; same account
   ```

   These logins are yours. They're stored on your computer and never in the project.

## Step 3: Configure

1. Open `deploy/deploy.sh` and set `PROJECT_ID="YOUR-PROJECT-ID"` at the top.
   The other settings can stay as they are.
2. Open `.firebaserc` and replace `YOUR-FIREBASE-PROJECT-ID` with the same ID.
3. Don't change `serviceId` or `region` in `firebase.json` unless you also change
   `SERVICE` and `REGION` in the script. They must match.

## Step 4: Run the script

In Git Bash, from the project folder:

```bash
bash deploy/deploy.sh infra      # 10-15 min: APIs, database, secrets, service account
bash deploy/deploy.sh release    # 5-10 min: builds the image and starts it on Cloud Run
bash deploy/deploy.sh hosting    # 1 min: publishes https://YOUR-PROJECT.web.app
```

`infra` needs no input. It's safe to run again because it skips anything that
already exists. It **never** changes an existing secret, so
re-running it won't invalidate cookies or the database password.

For later code changes, you only need `bash deploy/deploy.sh release`. Firebase
Hosting doesn't need redeploying unless `firebase.json` changes.

**If something fails:**
- *"Permission denied" while building (Cloud Build can't push to Artifact Registry).*
  In newer projects Cloud Build runs as the Compute Engine default service account.
  Grant that account `Artifact Registry Writer`, `Logs Writer` and
  `Storage Object Viewer` in *IAM*, then run `release` again.
- *Firebase says the project isn't a Firebase project.* Run
  `firebase projects:addfirebase YOUR-PROJECT-ID` once.
- *The site shows "Service Unavailable" right after deploying.* Wait a minute. The
  first start loads the model and the demand data.

## Step 5: Verification checklist (do all of these before piloting)

- [ ] **Front door.** Open `https://YOUR-PROJECT.web.app` in Chrome on a desktop.
      The consent/welcome page loads over HTTPS.
- [ ] **Health.** `curl -i https://YOUR-PROJECT.web.app/healthz` returns `200`.
- [ ] **Desktop-only.** Open the URL on a phone. You should be turned away politely.
- [ ] **Cookie.** In Chrome DevTools (*Application*, then *Cookies*), the only session
      cookie is `__session`. Firebase Hosting drops every other cookie name, so if the
      game "forgets" you after the first click, this is the cause.
- [ ] **One full test session.** Use a participant ID you'll recognise, such as
      `TEST-PROD-01`, and play all practice and main weeks through to the completion
      screen. Then open the SONA study link once as a test participant and check
      that the participant ID arrives filled in from the URL.
- [ ] **Instance count.** In the Cloud Console go to *Cloud Run*, then
      *reliance-study*, then *Revisions*. The latest revision should show min 1, max 1.
- [ ] **The log replays exactly.** Connect to the production database through the
      Cloud SQL Auth Proxy (below) and run:

      ```bash
      export DATABASE_URL="postgresql://reliance_app:PASSWORD@127.0.0.1:5432/reliance"
      python src/interface/verify_log.py --participant TEST-PROD-01
      ```

      It must report that every logged week reproduces. (This needs the Postgres
      support in `src/interface/`. If `verify_log.py` still only accepts `--db` with
      a SQLite file, export the tables first, as described below.)
- [ ] **Clean up.** Withdraw or delete the test participants before real collection
      starts (`python src/interface/withdraw.py --participant TEST-PROD-01`, run
      through the proxy), or record their IDs so you can exclude them in analysis.

### Connecting to the production database from your computer

1. Download the Cloud SQL Auth Proxy for Windows:
   <https://cloud.google.com/sql/docs/postgres/connect-auth-proxy#install>
   and save it as `cloud-sql-proxy.exe` in the project folder. It's git-ignored.
2. Log in once for application credentials: `gcloud auth application-default login`
3. Start the proxy and leave the window open:

   ```bash
   ./cloud-sql-proxy.exe --port 5432 YOUR-PROJECT-ID:europe-west1:reliance-db
   ```

4. Get the password from the secret. Treat it as confidential:

   ```bash
   gcloud secrets versions access latest --secret=DATABASE_URL
   # postgresql://reliance_app:THE-PASSWORD@/reliance?host=/cloudsql/...
   ```

   For local tools, use `postgresql://reliance_app:THE-PASSWORD@127.0.0.1:5432/reliance`.

## Exporting the data for analysis

With the proxy running, and `psql` from the PostgreSQL installer
(<https://www.postgresql.org/download/windows/>, *Command Line Tools* only):

```bash
export PGURL="postgresql://reliance_app:THE-PASSWORD@127.0.0.1:5432/reliance"
mkdir -p exports/$(date +%F)
for t in participant trial block_measure event_log screening; do
  psql "$PGURL" -c "\copy $t TO 'exports/$(date +%F)/$t.csv' WITH CSV HEADER"
done
# A complete, restorable snapshot as well:
pg_dump "$PGURL" --format=custom --file="exports/$(date +%F)/reliance.dump"
```

The CSVs have the same table and column layout as the local SQLite exports
(`db.export_csv`), so the analysis scripts work unchanged. `exports/` is
git-ignored. Store it only where your ethics approval allows, because it contains
participant IDs (SONA identification codes).

## Backups and restore

- **Automatic.** Cloud SQL takes a daily backup (kept 30 days) and keeps 7 days of
  transaction logs, so you can recover to any second in the last week.
- **Your own copy.** Run the export above at the end of every collection day. It's
  the copy that doesn't depend on Google.
- **Point-in-time recovery**, for example after an accidental deletion: never restore
  over the live database. Clone to a new instance at a moment just before the problem:

  ```bash
  gcloud sql instances clone reliance-db reliance-db-restore \
    --point-in-time "2026-10-25T14:00:00Z"
  ```

  Then inspect or export from `reliance-db-restore` through the proxy, and delete the
  clone when you're done (`gcloud sql instances delete reliance-db-restore`). The clone
  is billed while it exists.
- **Restore a daily backup** (this overwrites the target instance):
  `gcloud sql backups list --instance=reliance-db`, then
  `gcloud sql backups restore BACKUP_ID --restore-instance=reliance-db-restore`.
- The instance has deletion protection turned on. Deleting it takes a deliberate
  extra step.

## Rolling back a bad release

Every release is a numbered Cloud Run revision. To send all traffic back to the
previous one (about 10 seconds):

```bash
gcloud run revisions list --service=reliance-study --region=europe-west1
gcloud run services update-traffic reliance-study --region=europe-west1 \
  --to-revisions=reliance-study-00007-abc=100
```

As with a deploy, switching revisions restarts the app and drops in-memory games.
Participants mid-session will have to reload and re-enter their ID. Don't do this
during collection unless the app is actually broken.

## Deploy freeze: 19 October to 1 November 2026

During data collection **nothing is deployed**. Any deploy or rollback replaces the
only running instance, and any code change mid-study is a confound. To enforce this,
create the freeze file before collection starts:

```bash
echo "Data collection 19 Oct - 1 Nov 2026. No deploys." > deploy/FREEZE
```

`deploy/deploy.sh` refuses to run while that file exists. Delete it after the last
session on 1 November. If a real emergency (the app is down) forces a change, write
down what, when and why in the study log before deleting the file.

Also during collection:
- Don't edit secrets or Cloud Run settings in the console.
- Check the Cloud Run *Logs* tab once a day for errors.
- Run the data export at the end of each day.

## Rough monthly cost (europe-west1, list prices, late 2026)

| Item | Approx. per month |
|---|---|
| Cloud Run: 1 always-on instance (1 vCPU, 1 GiB), mostly idle | USD 10-20 |
| Cloud SQL: `db-g1-small`, 10 GB SSD, backups | USD 27-32 |
| Cloud SQL alternative: `db-f1-micro` (set `SQL_TIER`; max ~25 connections) | USD 10-12 |
| Artifact Registry, Secret Manager, Cloud Build, Hosting | < USD 2 |
| **Total** | **about USD 40-55** (about USD 25-35 with `db-f1-micro`) |

These are estimates. Confirm them in the Google Cloud pricing calculator and with
your budget alert. To stop paying after the study, first take a final export, then:

```bash
gcloud run services update reliance-study --region=europe-west1 --min-instances=0
gcloud sql instances patch reliance-db --activation-policy=NEVER   # stop the DB (storage still billed)
```

Delete the resources only when your data-retention plan allows it.

## Data protection

- **Region.** Everything the script creates (Cloud Run, Cloud SQL, secrets, images)
  is in `europe-west1`, inside the EU. Firebase Hosting is a global CDN, but it only
  forwards requests and doesn't store study data.
- **Personal data.** Participant IDs taken from SONA are pseudonymous but still
  **personal data** under the GDPR, because the SONA pool can link them to a
  student. They're stored in Cloud SQL, together with IP-derived request logs in
  Cloud Logging.
- **Processor.** Google acts as a data processor here. Your ethics approval and
  participant information sheet must cover storing data with Google Cloud (Google
  Ireland Ltd.) in the EU. Accept the Cloud Data Processing Addendum in the console
  (*IAM & Admin*, then *Settings*, then *Data processing terms*) if your institution
  requires it.
- **Access.** Only the runtime service account (`reliance-run`) can read the two
  secrets and connect to the database. Keep the number of people with *Owner* or
  *Editor* on the project to a minimum.
- **Withdrawal.** Participant withdrawal requests are handled with
  `src/interface/withdraw.py` through the proxy, as in local use.
