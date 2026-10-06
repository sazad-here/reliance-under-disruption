"""
Live demo: the participant app, running the live LightGBM risk model and
error design v2, played end to end in a real browser by a scripted participant.

Writes to demo/output/: screenshots of every stage, a video of the session,
and demo_weeks.csv -- one row per main week with the state, the AI's risk and
recommendation, whether the AI was right, the participant's own and final
order, and what happened. Uses its own database (demo/output/demo.sqlite3),
never the collection file.

Run:  python demo/run_demo.py            (hidden browser, fast)
      python demo/run_demo.py --headed   (visible browser window, slowed so you can watch)
      python demo/run_demo.py --slow     (visible, human speed: typed orders, the practice
                                           tour clicked through, the 3D week animation shown)
"""
from __future__ import annotations

import os
import shutil
import sqlite3
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "demo" / "output"
sys.path.insert(0, str(ROOT / "src"))
# Remove only this script's own outputs. demo/output also holds other files
# (e.g. play_yourself.sqlite3, the database of a manually played session),
# which must never be wiped by a demo run.
OUT.mkdir(parents=True, exist_ok=True)
shutil.rmtree(OUT / "shots", ignore_errors=True)
for old in list(OUT.glob("demo.sqlite3*")) + [OUT / "demo_session.webm", OUT / "demo_weeks.csv"]:
    old.unlink(missing_ok=True)
(OUT / "shots").mkdir(parents=True)

import interface.db as db  # noqa: E402
db.DEFAULT_DB = OUT / "demo.sqlite3"
from interface import app as app_module  # noqa: E402

PORT = 8791
SLOW = "--slow" in sys.argv
HEADED = "--headed" in sys.argv or SLOW
PID = "DEMO-01"
threading.Thread(target=lambda: app_module.app.run(port=PORT, threaded=True), daemon=True).start()
time.sleep(1.5)

from playwright.sync_api import sync_playwright  # noqa: E402

shot_n = [0]


def shot(page, name):
    shot_n[0] += 1
    page.screenshot(path=str(OUT / "shots" / ("%02d_%s.png" % (shot_n[0], name))), full_page=True)


def own_order(page):
    """A plausible, imperfect participant: cover ~2.5 weeks of last demand."""
    cells = page.query_selector_all("#state .cell .v")
    vals = [c.inner_text().replace(",", "") for c in cells]
    demand, stock, on_order, backorders = (int(vals[0]), int(vals[1]), int(vals[2]), int(vals[4]))
    position = stock + on_order - backorders
    return int(max(0, round(2.5 * demand - position + 140)))


def tour(page):
    """Practice week 1: read each guided-tour card, then press "Got it"."""
    for _ in range(3):
        if not page.is_visible("#coachNext"):
            return
        page.wait_for_timeout(2600)
        page.click("#coachNext")
        page.wait_for_timeout(500)


errors = []
with sync_playwright() as pw:
    browser = pw.chromium.launch(headless=not HEADED, slow_mo=(260 if SLOW else 120) if HEADED else 0,
                                 args=["--window-size=1480,1000"] if HEADED else [])
    ctx = browser.new_context(viewport={"width": 1440, "height": 900},
                              record_video_dir=str(OUT), record_video_size={"width": 1440, "height": 900})
    page = ctx.new_page()
    page.on("pageerror", lambda e: errors.append(str(e)))

    page.goto("http://127.0.0.1:%d/?PROLIFIC_PID=%s" % (PORT, PID))
    page.wait_for_timeout(400); shot(page, "consent")
    if SLOW:
        page.wait_for_timeout(1500)
    page.check("#consent"); page.click("#next1")
    for sel, v in (("#num_1", "500"), ("#num_2", "10"), ("#num_3", "0.1")):
        if SLOW:
            page.type(sel, v, delay=160)
        else:
            page.fill(sel, v)
    page.wait_for_timeout(300); shot(page, "numeracy_screener")
    page.click("#go"); page.wait_for_url("**/task")
    if SLOW:
        page.wait_for_timeout(2500)     # let the 3D warehouse load and be seen

    sess = app_module._SESSIONS[PID]
    shown = set()
    for _ in range(400):
        if page.is_visible("#s1") and page.is_enabled("#s1go"):
            phase, idx = sess.phase, sess.trial_index
            q_own = min(own_order(page), sess.params.max_order)
            if SLOW:
                tour(page)
                page.type("#qown", str(q_own), delay=180); page.wait_for_timeout(600)
            else:
                page.fill("#qown", str(q_own))
            if phase == "main" and "main_state" not in shown:
                shown.add("main_state"); shot(page, "main_week_state_panel")
            page.click("#s1go")
            try:
                page.wait_for_selector("#s2:not(.hide)", timeout=5000)
            except Exception:
                raise SystemExit("stage 1 rejected at %s week %d (own order %d): %s"
                                 % (phase, idx, q_own, page.inner_text("#s1err")))
            q_ai, risk = int(page.inner_text("#qai")), int(page.inner_text("#risk"))
            correct = sess.schedule[idx]
            if HEADED:
                page.locator("#s2ai").scroll_into_view_if_needed(); page.wait_for_timeout(900)
            if phase == "main" and idx == 12 and "dark" not in shown:
                shown.add("dark"); page.emulate_media(color_scheme="dark"); page.wait_for_timeout(200)
                shot(page, "dark_mode_ai_suggestion"); page.emulate_media(color_scheme="light")
            if phase == "main":
                key = "main_ai_correct" if correct else "main_ai_error"
                if key not in shown:
                    shown.add(key); shot(page, key)
            # accept when close to own view; keep own when the AI looks far off;
            # otherwise meet in the middle
            gap = abs(q_ai - q_own)
            final = q_ai if gap <= 40 else (q_own if gap > 110 else int(round((q_ai + q_own) / 2)))
            final = min(max(final, 0), sess.params.max_order)
            if SLOW:
                tour(page); page.wait_for_timeout(1200)
                page.type("#qfinal", str(final), delay=180); page.wait_for_timeout(600)
            else:
                page.fill("#qfinal", str(final))
            page.click("#s2go")
            page.wait_for_selector("#out:not(.hide)")
            if SLOW:
                page.locator("#scene3dCard").scroll_into_view_if_needed() if page.is_visible("#scene3dCard") else None
                page.wait_for_timeout(3900)     # the 3D week (3.4 s): truck, racks, conveyor
                tour(page)
            elif HEADED:
                page.wait_for_timeout(700)
            if phase == "main" and page.inner_text("#outalert").strip() and "stockout" not in shown:
                shown.add("stockout"); shot(page, "outcome_stockout")
            if phase == "practice" and idx == 0:
                shot(page, "practice_outcome")
            page.click("#next")
        elif page.is_visible("#gate"):
            shot(page, "practice_complete"); page.click("#gatego")
        elif page.is_visible("#quiz"):
            for q in page.query_selector_all("#quizItems .q"):
                q.query_selector_all("input")[3].check()
            if "quiz" not in shown:
                shown.add("quiz"); shot(page, "questionnaire")
            page.click("#quizgo")
        elif page.is_visible("#done"):
            break
        page.wait_for_timeout(40)

    page.fill("#perceived", "75"); page.click("#donego")
    page.wait_for_selector("#debrief:not(.hide)"); page.wait_for_timeout(300)
    shot(page, "debrief")
    ctx.close(); browser.close()

video = sorted(OUT.glob("*.webm"))
if video:
    video[0].rename(OUT / "demo_session.webm")

con = sqlite3.connect(db.DEFAULT_DB)
t = pd.read_sql("SELECT * FROM trial WHERE participant_id=? AND phase='main' ORDER BY trial_index",
                con, params=(PID,))
t["stockout_next_2"] = [int(t.stockout.iloc[i + 1:i + 3].max()) if i + 1 < len(t) else np.nan
                        for i in range(len(t))]
cols = ["trial_index", "period", "demand", "inventory", "on_order", "backorders", "disruption",
        "placement", "ai_correct", "q_star", "q_ai", "ai_risk_pct", "q_own", "q_final",
        "response_class", "woa", "cost", "stockout", "stockout_next_2"]
t[cols].to_csv(OUT / "demo_weeks.csv", index=False)
cond = con.execute("SELECT condition FROM participant WHERE participant_id=?", (PID,)).fetchone()[0]

print("condition:", cond, "| design:", app_module.ADVICE_DESIGN, "| page errors:", errors or "none")
print("AI right on %d of %d weeks | errors at weeks %s" % (
    t.ai_correct.sum(), len(t), t.loc[t.ai_correct == 0, "trial_index"].tolist()))
print("responses:", t.response_class.value_counts().to_dict(), "| total cost %.0f | stockout weeks %d"
      % (t.cost.sum(), t.stockout.sum()))
print(t[["trial_index", "disruption", "placement", "ai_correct", "q_star", "q_ai", "ai_risk_pct",
         "q_own", "q_final", "response_class", "stockout"]].to_string(index=False))
print("outputs ->", OUT)
