"""
Narrated end-to-end demo video of the study: slides that explain how the AI was built and
how the experiment is designed, then a live session in the real participant app, then the
log verification and the analysis plan. Every step is spoken and captioned.

Pipeline (video-editing skill): capture (Playwright records the real app) -> structure
(the SCENES list below) -> voice (Windows SAPI, offline, voice "Microsoft Zira") ->
FFmpeg (narration mixed at each scene's start, captions burned in, H.264/AAC MP4).

Each scene's narration is synthesised FIRST; the browser then holds every scene until its
narration has finished, so picture and voice stay in sync without manual editing.

Run:  python demo/narrated/make_narrated_demo.py
Out:  demo/narrated/output/Reliance_Under_Disruption_demo.mp4  (+ captions.srt, narration.wav,
      transcript.md). Only this script's own files in that folder are replaced.
"""
from __future__ import annotations

import base64
import json
import os
import re
import subprocess
import sys
import threading
import time
import wave
from pathlib import Path

import imageio_ffmpeg

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "demo" / "narrated" / "output"
OUT.mkdir(parents=True, exist_ok=True)
for old in ["narrated.sqlite3", "narrated.sqlite3-wal", "narrated.sqlite3-shm", "raw.webm",
            "narration.wav", "captions.srt", "transcript.md", "Reliance_Under_Disruption_demo.mp4"]:
    (OUT / old).unlink(missing_ok=True)
for old in OUT.glob("scene_*.wav"):
    old.unlink()
for old in OUT.glob("*.webm"):
    old.unlink()

sys.path.insert(0, str(ROOT / "src"))
import interface.db as db  # noqa: E402

db.DEFAULT_DB = OUT / "narrated.sqlite3"
from interface import app as app_module  # noqa: E402
from interface.advice import placement_of  # noqa: E402

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
PORT, PID = 8792, "VIDEO-01"
W, H = 1280, 800
CAP_H = 130          # caption strip added under the 1280x800 recording
FIG = ROOT / "figures" / "paper"


# ------------------------------------------------------------------ narration script
NARR = {
    "title": "Welcome. This video walks through our study from beginning to end. We built a simulated "
             "supply chain, called a digital twin, and an AI assistant that advises people on how much "
             "stock to order. The question is how people decide whether to rely on that AI, and in "
             "particular whether it matters where in time the AI makes its mistakes.",
    "pipeline": "First, how the AI was built. The digital twin is calibrated on three open datasets: real "
                "Walmart demand from the M5 competition, delivery times from Olist, and disruption rates "
                "from DataCo. The twin generated fifty-two thousand simulated weeks, which trained a "
                "LightGBM model to predict the chance of a stockout within the next two weeks. The model "
                "adds up one hundred and ninety-four decision trees, and it is calibrated, so its "
                "percentage means what it says. On demand it never saw in training, it reaches an AUROC "
                "of zero point nine.",
    "design": "Second, the experiment design. Participants run the warehouse for thirty weeks. The AI's "
              "order suggestion is right on either eighty or sixty percent of the weeks, depending on the "
              "condition. Its errors are always under-orders, and they are placed on purpose: some on "
              "weeks where one bad order propagates into later stockouts, others on weeks where the "
              "system absorbs it. In held-out simulations, a propagating error causes three and a half "
              "times as many extra stockout weeks.",
    "twin": "Third, how the simulator works inside. Each week follows the same loop. The participant "
            "places an order. It arrives one to four weeks later, with the delay drawn at random from "
            "the Olist delivery data: one week about half the time, and four weeks about one time in "
            "twelve. Customer demand, a real Walmart sales series, is served from the stock on hand. "
            "Demand that cannot be met is not lost; it becomes unmet orders that must be filled first "
            "the following week. Every unit left on the shelf costs one, and every unit short costs "
            "five. On top of that, six disruptions hit on fixed weeks, the same for everyone: two "
            "supplier shortfalls, a demand spike, a transport delay, a longer lead time, and a loss of "
            "a quarter of the stock.",
    "liveai": "Fourth, how the AI's advice is made. It is computed live, every week, from that "
              "participant's own warehouse. The simulator first works out the ideal order for the "
              "current state. On a correct week, the AI suggests an order close to that ideal. On an "
              "error week, it suggests ninety to one hundred and twenty units too few. Only one thing is "
              "fixed in advance: which weeks are correct and which are errors. That keeps the accuracy "
              "at exactly eighty or sixty percent for everyone, and lets us control where the errors "
              "fall. The risk percentage comes from the LightGBM model running on the participant's own "
              "weekly history. The model is frozen. It never learns from participants, so every "
              "participant faces the same AI.",
    "consent":"Now the live simulator. A participant arrives from Prolific, reads the information sheet, "
               "and gives consent. The study runs only on desktop computers; phones and small windows are "
               "politely turned away.",
    "screener": "Next is a three-question numeracy screener. For example: if a fair die is rolled one "
                "thousand times, how often does it land on an even number? Five hundred. Participants "
                "must read a stockout probability, so anyone with fewer than two correct answers does not "
                "continue: they see a thank-you screen, and only their anonymous score is logged. The "
                "screener runs before a condition is assigned, so it cannot unbalance the groups.",
    "state": "This is the task screen. At the top is the thirty-week timeline. The panel shows this "
             "week's warehouse: last week's demand, stock on hand, stock already on order, the lead "
             "time, unmet orders, the supplier and transport status, and the cost so far.",
    "own": "Every week has two stages. First, the participant enters their own order, before seeing any "
           "advice. Once submitted, it locks and cannot be changed. This unaided order is the baseline "
           "that lets us measure how much the AI changes the decision.",
    "advice": "Only now does the AI assistant appear. It suggests an order, and it shows the live model's "
              "estimated chance of running out in the next two weeks, as a percentage and a risk level. "
              "The assistant is clearly labelled as a machine, and it reminds the participant that it is "
              "not always right.",
    "final": "The participant then enters a final order: keep their own, take the suggestion, or choose "
             "any other number. Keeping and taking cost the same effort; there is no one-click accept "
             "button. The twin then runs the week and shows what happened: the cost, and whether any "
             "customer orders went unmet.",
    "practice": "Five practice weeks let participants learn the controls. We move through the rest of the "
                "practice quickly.",
    "main": "The main phase starts from a fresh warehouse, and these thirty weeks count toward the "
            "results. The demand is a real Walmart sales series, identical for every participant. We move "
            "forward to the AI's first scripted error on a propagating week.",
    "error": "This week, the AI's suggestion is one of its scripted errors. It sits on a propagating week, "
             "so accepting it starts a shortage that can spread into the following weeks. The risk "
             "estimate is computed on the state the AI believes, so the number agrees with the wrong "
             "order, as a real model's mistake would. Our demo participant accepts the suggestion.",
    "consequence": "Over the next weeks, the effect of that single decision appears in the outcome panel "
                   "as unmet orders and higher cost. This carry-over of consequences from week to week "
                   "is exactly what the study measures.",
    "quiz": "After every ten weeks, participants rate their trust in the assistant and their workload. "
            "Measuring trust three times shows whether it drifts apart between the eighty and sixty "
            "percent conditions.",
    "forward": "We now fast-forward through the remaining weeks.",
    "perceived": "At the end, participants estimate how often the AI was right. This checks that the two "
                 "reliability conditions were actually perceived differently.",
    "debrief": "The debrief then discloses everything: the suggestions were scripted, how many were "
               "correct, and where the errors were placed. Participants may still withdraw and have all "
               "of their data deleted.",
    "complete": "At the bottom, the participant chooses to keep their data in the study, or to delete "
                "all of it. Then the Prolific completion code appears, which confirms the session so "
                "they are paid. Payment never depends on performance. In this demo the code is a "
                "placeholder; the real one is set when the Prolific study is created.",
    "data": "Where does the data go? Every click is saved on the study server the moment it happens, "
            "in a secured database in a European region. Participants are identified only by their "
            "Prolific ID. If the connection drops, the participant continues exactly where they "
            "stopped. The data is never fed back into the AI model. It is only read after data "
            "collection closes, by the pre-registered analysis scripts, and anyone can still ask for "
            "their data to be deleted.",
    "verify": "Every session can also be checked. Our verification script replays the whole "
              "session from the log. The simulator reproduces every week exactly, and the risk shown on "
              "screen matches the model recomputed offline. No check fails. The one warning is expected: "
              "the demo answers faster than any person could, which is exactly the click-through pattern "
              "our exclusion rule is designed to catch.",
    "quality": "Before the analysis, some sessions are set aside, using rules fixed in advance. A "
               "participant is excluded if they did not finish; if more than a fifth of their own orders "
               "took under half a second, which means clicking through; if they typed the same own order "
               "every week; if they withdrew; or if a technical fault stopped the task from displaying "
               "properly. A browser error counts as such a fault only if it happened during the main "
               "weeks and the log no longer replays exactly. Otherwise the session is kept and flagged.",
    "measures": "Three things are measured every week. Weight of advice: how far the participant moved "
                "from their own order toward the AI's. If they planned one hundred, the AI said two "
                "hundred, and they ordered one hundred and fifty, they moved halfway, a weight of zero "
                "point five. Appropriate reliance: following the AI when it is right, and not when it is "
                "wrong. And decision error: how far the final order was from the ideal one.",
    "hypotheses": "The study tests four predictions, registered before any data is collected. One: people "
                  "follow the eighty percent AI more than the sixty percent AI. Two: trust in the two "
                  "groups drifts apart over the three questionnaires. Three, the key question: after a "
                  "mistake that causes lasting damage, people rely on the AI less than after an equally "
                  "large mistake that is quietly absorbed. Four: the sixty percent group makes worse, "
                  "costlier decisions, and the gap grows after the damaging mistakes. Mixed-effects "
                  "models compare each person against their own baseline, because some people simply "
                  "follow advice more than others.",
    "plan": "The plan is to recruit eighty participants, so that about seventy remain after "
            "exclusions: thirty-five per condition. That number is confirmed with the supervisor before "
            "the pre-registration is submitted. A pilot runs on the thirteenth of October, data "
            "collection from the nineteenth of October to the first of November, twenty twenty-six, "
            "and the results go to AHFE twenty twenty-seven.",
    "close": "That is the whole study: open data and model building, the simulator and the live AI, "
             "each decision a participant makes, where the data goes, how it is checked, and the "
             "analysis that follows. Thank you for watching.",
}


# ------------------------------------------------------------------ voice (Windows SAPI, offline)
def synthesise():
    jobs = [{"text": t, "path": str(OUT / ("scene_%s.wav" % k))} for k, t in NARR.items()]
    (OUT / "tts_jobs.json").write_text(json.dumps(jobs), encoding="utf-8")
    ps = OUT / "tts.ps1"
    ps.write_text(
        "Add-Type -AssemblyName System.Speech\n"
        "$jobs = Get-Content -Raw -Encoding UTF8 '%s' | ConvertFrom-Json\n"
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer\n"
        "$s.SelectVoice('Microsoft Zira Desktop'); $s.Rate = 0; $s.Volume = 100\n"
        "foreach ($j in $jobs) { $s.SetOutputToWaveFile($j.path); $s.Speak($j.text); $s.SetOutputToNull() }\n"
        % str(OUT / "tts_jobs.json"), encoding="utf-8")
    subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(ps)], check=True)
    durs = {}
    for k in NARR:
        with wave.open(str(OUT / ("scene_%s.wav" % k))) as w:
            durs[k] = w.getnframes() / w.getframerate()
    (OUT / "tts_jobs.json").unlink(); ps.unlink()
    return durs


# ------------------------------------------------------------------ slides
def img64(p):
    return "data:image/png;base64," + base64.b64encode(Path(p).read_bytes()).decode()


def slide(title, body_html, kicker="Reliance Under Disruption · AHFE 2027"):
    return """<html><head><style>
    body{margin:0;width:%dpx;height:%dpx;background:#fbfbf9;font-family:Arial,Helvetica,sans-serif;color:#1a1a19;
         display:flex;flex-direction:column;padding:34px 56px;box-sizing:border-box}
    .k{font-size:15px;letter-spacing:.08em;text-transform:uppercase;color:#2a78d6;font-weight:bold}
    h1{font-size:40px;margin:10px 0 18px}
    .b{flex:1;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:14px}
    img{max-width:100%%;max-height:520px;box-shadow:0 2px 10px rgba(0,0,0,.12);background:white}
    ul{font-size:24px;line-height:1.5;margin:0;align-self:flex-start}
    pre{font-size:16px;background:#1a1a19;color:#e8f5e8;padding:18px 22px;border-radius:8px;align-self:stretch;
        line-height:1.35;white-space:pre-wrap}
    .big{font-size:52px;font-weight:bold;text-align:center;line-height:1.2}
    .sub{font-size:24px;color:#52514e;text-align:center}
    .row{display:flex;gap:14px;align-items:stretch;justify-content:center;align-self:stretch}
    .box{flex:1;background:white;border:2px solid #d6d4cc;border-radius:10px;padding:14px 16px;font-size:19px;line-height:1.35}
    .box b{display:block;font-size:21px;margin-bottom:6px;color:#1a1a19}
    .box.blue{border-color:#2a78d6}.box.amber{border-color:#e8a33d}.box.grey{border-color:#8a8983;background:#f1f0ec}
    .arr{align-self:center;font-size:30px;color:#8a8983}
    .note{font-size:19px;color:#52514e;align-self:stretch;text-align:center}
    table.t{border-collapse:collapse;font-size:18px;align-self:stretch}
    table.t td{border-top:1px solid #d6d4cc;padding:7px 10px}
    ol{font-size:22px;line-height:1.45;margin:0;align-self:flex-start}
    </style></head><body><div class="k">%s</div><h1>%s</h1><div class="b">%s</div></body></html>""" % (
        W, H, kicker, title, body_html)


# ------------------------------------------------------------------ recording helpers
class Clock:
    def __init__(self):
        self.t0 = None
        self.marks = []   # (scene key, start seconds)

    def start(self):
        self.t0 = time.monotonic()

    def now(self):
        return time.monotonic() - self.t0


def scene(page, clock, key, durs, action=None, pad=0.7):
    """Run `action(until)` while this scene's narration plays; `until(f)` waits until
    fraction f of the narration has been spoken, so on-screen steps land on the words."""
    start = clock.now()
    clock.marks.append((key, start))

    def until(frac):
        left = start + frac * durs[key] - clock.now()
        if left > 0:
            page.wait_for_timeout(int(left * 1000))
    if action:
        action(until)
    left = durs[key] + pad - (clock.now() - start)
    if left > 0:
        page.wait_for_timeout(int(left * 1000))


def highlight(page, sel, on=True):
    page.evaluate("""([s, on]) => { const e = document.querySelector(s); if (!e) return;
        e.style.outline = on ? '4px solid #e8a33d' : ''; e.style.outlineOffset = on ? '4px' : '';
        if (on) e.scrollIntoView({block: 'center', behavior: 'smooth'}); }""", [sel, on])


def own_order(page):
    vals = [c.inner_text().replace(",", "") for c in page.query_selector_all("#state .cell .v")]
    demand, stock, on_order, backorders = int(vals[0]), int(vals[1]), int(vals[2]), int(vals[4])
    return int(max(0, round(2.5 * demand - (stock + on_order - backorders) + 140)))


def type_own(page, sess, slow=False):
    """Stage 1: type the demo participant's own order (a plausible, imperfect rule)."""
    q_own = min(own_order(page), sess.params.max_order)
    if slow:
        page.type("#qown", str(q_own), delay=140)
    else:
        page.fill("#qown", str(q_own))
    return q_own


def submit_own(page):
    """Lock the own order; returns the AI's suggested order once the advice appears."""
    page.click("#s1go")
    page.wait_for_selector("#s2:not(.hide)")
    return int(page.inner_text("#qai"))


def decide(sess, q_own, q_ai):
    """The demo participant: take close advice, keep own if far off, else meet in the middle."""
    gap = abs(q_ai - q_own)
    final = q_ai if gap <= 40 else (q_own if gap > 110 else int(round((q_ai + q_own) / 2)))
    return min(max(final, 0), sess.params.max_order)


def finish_week(page, final, slow=False):
    """Stage 2: enter (unless already typed) and confirm the final order."""
    if final is not None:
        if slow:
            page.type("#qfinal", str(final), delay=140)
        else:
            page.fill("#qfinal", str(final))
    page.click("#s2go")
    page.wait_for_selector("#out:not(.hide)")


def answer_quiz(page):
    page.wait_for_timeout(400)     # the page re-renders between the trust and workload forms
    for q in page.query_selector_all("#quizItems .q"):
        q.query_selector_all("input")[3].check()
    page.click("#quizgo")
    page.wait_for_timeout(400)


def advance(page):
    """Move past whatever non-trial screen is showing (gate / quiz)."""
    for _ in range(20):
        if page.is_visible("#s1") and page.is_enabled("#s1go"):
            return
        if page.is_visible("#gate"):
            page.click("#gatego")
        elif page.is_visible("#quiz"):
            answer_quiz(page)
        elif page.is_visible("#done"):
            return
        page.wait_for_timeout(120)


# ------------------------------------------------------------------ the recording
def record(durs):
    # a visible placeholder so the debrief shows where the Prolific code appears
    app_module.COMPLETION_CODE = "DEMO-CODE"
    threading.Thread(target=lambda: app_module.app.run(port=PORT, threaded=True), daemon=True).start()
    time.sleep(1.5)
    from playwright.sync_api import sync_playwright

    clock = Clock()
    verify_text = ""
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        ctx = browser.new_context(viewport={"width": W, "height": H}, record_video_dir=str(OUT),
                                  record_video_size={"width": W, "height": H})
        page = ctx.new_page()
        clock.start()

        # --- slides: what the study is and how the AI was built
        page.set_content(slide("", '<div class="big">Reliance Under Disruption</div>'
                                   '<div class="sub">A human–AI decision experiment in a digital-twin supply chain</div>'
                                   '<div class="sub" style="font-size:20px;margin-top:30px">Narrated walkthrough of the live simulator</div>'))
        scene(page, clock, "title", durs)
        page.set_content(slide("1 · How the AI model was built",
                               '<img src="%s">' % img64(FIG / "main_text" / "Figure_model_pipeline.png")))
        scene(page, clock, "pipeline", durs)
        page.set_content(slide("2 · Where the AI's errors are placed",
                               '<img src="%s">' % img64(FIG / "main_text" / "Figure2_error_placement.png")))
        scene(page, clock, "design", durs)
        page.set_content(slide("3 · How the simulator works inside", """
          <div class="row">
            <div class="box blue"><b>1 · Order</b>The participant places this week's order.</div><div class="arr">→</div>
            <div class="box"><b>2 · Delivery</b>Arrives 1–4 weeks later (Olist):<br>1 wk 52% · 2 wk 29%<br>3 wk 11% · 4 wk 8%</div><div class="arr">→</div>
            <div class="box"><b>3 · Demand</b>Real Walmart (M5) sales, served from stock on hand.</div><div class="arr">→</div>
            <div class="box amber"><b>4 · Carry-over</b>Unmet demand becomes unmet orders, filled first next week.</div>
          </div>
          <div class="row"><div class="box"><b>Cost each week</b>1 per unit left on the shelf · 5 per unit short</div></div>
          <table class="t"><tr><td><b>Fixed disruptions (same for everyone)</b></td><td>week 6: supplier ships 55%</td><td>week 9: demand +80%</td><td>week 13: transport +2 weeks</td></tr>
          <tr><td></td><td>week 17: lead time +2 weeks</td><td>week 21: 25% of stock lost</td><td>week 26: supplier ships 40%</td></tr></table>"""))
        scene(page, clock, "twin", durs)
        page.set_content(slide("4 · How the AI's advice is made each week", """
          <div class="row">
            <div class="box blue"><b>Live, every week</b>From this participant's own warehouse:<br>• ideal order Q* = order-up-to level − inventory position<br>
              • correct week: Q* ± 30 units<br>• error week: 90–120 units too few<br>• risk %: LightGBM on their own weekly history</div>
            <div class="box amber"><b>Fixed in advance</b>Only <i>which</i> weeks are correct:<br>• 24 of 30 (80%) or 18 of 30 (60%)<br>
              • errors on planned weeks: propagating or absorbed<br><br>So accuracy is exact and error timing is controlled.</div>
            <div class="box grey"><b>Frozen model</b>The model never learns from participants.<br><br>Everyone faces the same AI, from the first participant to the last.</div>
          </div>"""))
        scene(page, clock, "liveai", durs)

        # --- the live app. Rule: a page change happens at the START of the scene that
        # narrates the new page; within a scene, steps land on the words via until().
        page.goto("http://127.0.0.1:%d/?PROLIFIC_PID=%s" % (PORT, PID))
        page.wait_for_timeout(500)

        def consent(until):
            highlight(page, "#consent")
            until(0.55)
            page.check("#consent")
        scene(page, clock, "consent", durs, consent)

        def screener(until):
            highlight(page, "#consent", False)
            page.click("#next1")
            page.wait_for_timeout(500)
            for frac, (sel, v) in zip((0.12, 0.45, 0.62), (("#num_1", "500"), ("#num_2", "10"), ("#num_3", "0.1"))):
                until(frac)
                highlight(page, sel)
                page.type(sel, v, delay=180)
                highlight(page, sel, False)
        scene(page, clock, "screener", durs, screener)

        box = {}

        def state(until):
            page.click("#go")
            page.wait_for_url("**/task")
            page.wait_for_timeout(700)
            highlight(page, "#timeline")
            until(0.3)
            highlight(page, "#timeline", False)
            highlight(page, "#state")
        scene(page, clock, "state", durs, state)
        sess = app_module._SESSIONS[PID]

        def own(until):
            highlight(page, "#state", False)
            highlight(page, "#s1")
            box["q_own"] = type_own(page, sess, slow=True)
            until(0.72)
            box["q_ai"] = submit_own(page)
        scene(page, clock, "own", durs, own)

        def advice(until):
            highlight(page, "#s1", False)
            highlight(page, "#s2ai")
        scene(page, clock, "advice", durs, advice)

        def final(until):
            highlight(page, "#s2ai", False)
            highlight(page, "#s2go")
            page.type("#qfinal", str(box["q_ai"]), delay=140)
            until(0.55)
            highlight(page, "#s2go", False)
            finish_week(page, None)
            highlight(page, "#out")
        scene(page, clock, "final", durs, final)

        def practice(until):
            highlight(page, "#out", False)
            page.click("#next")
            advance(page)
            while sess.phase == "practice" and page.is_visible("#s1"):
                q_own = type_own(page, sess)
                q_ai = submit_own(page)
                finish_week(page, decide(sess, q_own, q_ai))
                page.wait_for_timeout(250)
                page.click("#next")
                page.wait_for_timeout(150)
                if page.is_visible("#gate"):
                    break
        scene(page, clock, "practice", durs, practice)

        # the first scripted AI error that sits on a propagating week
        target = next(i for i, ok in enumerate(sess.main_schedule)
                      if not ok and placement_of(i, sess.design) == "propagating")

        def main_phase(until):
            advance(page)   # the practice -> main gate
            page.wait_for_timeout(1500)
            while sess.trial_index < target:
                q_own = type_own(page, sess)
                q_ai = submit_own(page)
                finish_week(page, decide(sess, q_own, q_ai))
                page.wait_for_timeout(300)
                page.click("#next")
                advance(page)
        scene(page, clock, "main", durs, main_phase)

        def error_week(until):
            advance(page)
            type_own(page, sess, slow=True)
            box["q_ai"] = submit_own(page)
            highlight(page, "#s2ai")
            until(0.82)
            page.type("#qfinal", str(box["q_ai"]), delay=140)   # the demo participant accepts the error
        scene(page, clock, "error", durs, error_week)

        def consequence(until):
            highlight(page, "#s2ai", False)
            finish_week(page, None)
            highlight(page, "#out")
            page.wait_for_timeout(2200)
            for _ in range(3):
                highlight(page, "#out", False)
                page.click("#next")
                advance(page)
                if not page.is_visible("#s1"):
                    break
                q_own = type_own(page, sess)
                q_ai = submit_own(page)
                finish_week(page, decide(sess, q_own, q_ai))
                highlight(page, "#out")
                page.wait_for_timeout(1800)
            highlight(page, "#out", False)
        scene(page, clock, "consequence", durs, consequence)

        def to_quiz(until):
            page.click("#next")
            page.wait_for_timeout(350)
            for _ in range(60):
                if page.is_visible("#quiz"):
                    break
                if page.is_visible("#s1") and page.is_enabled("#s1go"):
                    q_own = type_own(page, sess)
                    q_ai = submit_own(page)
                    finish_week(page, decide(sess, q_own, q_ai))
                    page.click("#next")
                    page.wait_for_timeout(350)
                else:
                    page.wait_for_timeout(150)
            page.wait_for_timeout(500)
            highlight(page, "#quiz")
            for q in page.query_selector_all("#quizItems .q"):
                q.query_selector_all("input")[3].check()
                page.wait_for_timeout(350)
        scene(page, clock, "quiz", durs, to_quiz)

        def forward(until):
            highlight(page, "#quiz", False)
            page.click("#quizgo")
            page.wait_for_timeout(400)
            for _ in range(400):
                if page.is_visible("#done"):
                    break
                advance(page)
                if page.is_visible("#s1") and page.is_enabled("#s1go"):
                    q_own = type_own(page, sess)
                    q_ai = submit_own(page)
                    finish_week(page, decide(sess, q_own, q_ai))
                    page.click("#next")
                page.wait_for_timeout(60)
        scene(page, clock, "forward", durs, forward)

        def perceived(until):
            highlight(page, "#perceived")
            until(0.4)
            page.type("#perceived", "70", delay=200)
        scene(page, clock, "perceived", durs, perceived)

        def debrief(until):
            highlight(page, "#perceived", False)
            page.click("#donego")
            page.wait_for_selector("#debrief:not(.hide)")
            highlight(page, "#debrief")
            until(0.55)
            page.mouse.wheel(0, 500)
        scene(page, clock, "debrief", durs, debrief)

        def complete(until):
            highlight(page, "#debrief", False)
            highlight(page, "#dbKeep")
            until(0.3)
            page.click("#dbKeep")
            highlight(page, "#dbKeep", False)
            until(0.42)
            highlight(page, "#dbCodeBox")
        scene(page, clock, "complete", durs, complete)

        page.set_content(slide("5 · Where the data goes", """
          <div class="row">
            <div class="box blue"><b>Participant's browser</b>Each order, answer and response time is sent the moment it happens.</div><div class="arr">→</div>
            <div class="box"><b>Study server</b>Secured database, EU region.<br>Identified only by Prolific ID.<br>A dropped connection resumes exactly.</div><div class="arr">→</div>
            <div class="box amber"><b>After 1 November</b>Pre-registered analysis scripts read the database → results → paper.</div>
          </div>
          <div class="row">
            <div class="box grey"><b>✕ Not fed into the AI model</b>The model is frozen, so it never learns from participants.</div>
            <div class="box grey"><b>Withdrawal at any time</b>During the task, at the debrief, or later by message: every row is deleted.</div>
          </div>"""))
        scene(page, clock, "data", durs)

        # --- behind the scenes: the real verification of THIS session
        r = subprocess.run([sys.executable, str(ROOT / "src" / "interface" / "verify_log.py"),
                            "--db", str(db.DEFAULT_DB), "--participant", PID],
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        verify_text = r.stdout
        lines = [ln for ln in verify_text.splitlines() if ln.strip()]
        keep = [ln for ln in lines if ("replay" in ln or "risk" in ln or "checks," in ln)][:6]
        body = "<pre>$ python src/interface/verify_log.py --participant %s\n\n%s</pre>" % (
            PID, "\n".join(keep).replace("<", "&lt;"))
        page.set_content(slide("6 · Every session is verified", body))
        scene(page, clock, "verify", durs)

        page.set_content(slide("7 · Quality control: exclusions fixed in advance",
                               "<ol><li>Session not finished</li>"
                               "<li>More than 20% of own orders entered in under 0.5 s (clicking through)</li>"
                               "<li>The same own order every week</li>"
                               "<li>Withdrew from the study</li>"
                               "<li>Technical fault. A browser error counts only if it happened in the main weeks <b>and</b> the log no longer replays exactly; otherwise the session is kept and flagged</li></ol>"
                               '<div class="note">Failing the numeracy screener is not an exclusion: those people never enter the study.</div>'))
        scene(page, clock, "quality", durs)

        page.set_content(slide("8 · What is measured every week", """
          <div class="row">
            <div class="box blue"><b>Weight of advice</b>How far the final order moved toward the AI.<br><br>own 100 · AI 200 · final 150<br>→ moved halfway = <b style="display:inline">0.5</b></div>
            <div class="box"><b>Appropriate reliance</b>Follow the AI when it is right; keep your own when it is wrong.</div>
            <div class="box amber"><b>Decision error</b>|final order − ideal order|<br><br>ideal 180 · final 150 → error 30</div>
          </div>"""))
        scene(page, clock, "measures", durs)

        page.set_content(slide("9 · The four pre-registered predictions", """
          <table class="t" style="font-size:21px">
          <tr><td><b>H1</b></td><td>People follow the 80% AI more than the 60% AI.</td></tr>
          <tr><td><b>H2</b></td><td>Trust in the two groups drifts apart over the three questionnaires.</td></tr>
          <tr style="background:#fff4e0"><td><b>H3 · key</b></td><td>After a mistake that causes <b style="display:inline">lasting damage</b>, reliance drops more than after an equally large mistake that is <b style="display:inline">absorbed</b>.</td></tr>
          <tr><td><b>H4</b></td><td>The 60% group makes worse, costlier decisions, and the gap grows after damaging mistakes.</td></tr></table>
          <div class="note">Mixed-effects models compare each person against their own baseline · Holm correction across tests</div>"""))
        scene(page, clock, "hypotheses", durs)

        page.set_content(slide("10 · Sample and timeline", """
          <div class="row">
            <div class="box blue"><b>Sample</b>Recruit 80 → about 70 analysed<br>35 per condition (80% / 60%)<br><br><span style="color:#52514e">To be confirmed with the supervisor before registration</span></div>
            <div class="box"><b>Timeline</b>Pilot: 13 Oct 2026<br>Collection: 19 Oct – 1 Nov 2026<br>Abstract: by 4 Dec 2026<br>Conference: AHFE 2027, Málaga</div>
          </div>"""))
        scene(page, clock, "plan", durs)

        page.set_content(slide("", '<div class="big">Thank you</div>'
                                   '<div class="sub">Reliance Under Disruption · digital twin · live LightGBM risk model · pre-registered experiment</div>'))
        scene(page, clock, "close", durs, pad=1.5)
        total = clock.now()
        ctx.close()
        browser.close()

    raw = next(OUT.glob("*.webm"))
    raw.rename(OUT / "raw.webm")
    return clock.marks, total, verify_text


# ------------------------------------------------------------------ captions + mix + encode
def srt_time(s):
    ms = int(round(s * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    sec, ms = divmod(ms, 1000)
    return "%02d:%02d:%02d,%03d" % (h, m, sec, ms)


def captions(marks, durs):
    out, n = [], 1
    for key, start in marks:
        text = NARR[key]
        parts = [p.strip() for p in re.split(r"(?<=[.:?])\s+", text) if p.strip()]
        chunks = []
        for p in parts:           # keep captions at most ~95 characters
            while len(p) > 95:
                cut = p.rfind(",", 0, 95)
                cut = cut if cut > 30 else p.rfind(" ", 0, 95)
                chunks.append(p[:cut + 1].strip()); p = p[cut + 1:].strip()
            chunks.append(p)
        total = sum(len(c) for c in chunks)
        t = start
        for c in chunks:
            d = durs[key] * len(c) / total
            out.append("%d\n%s --> %s\n%s\n" % (n, srt_time(t), srt_time(t + d), c))
            n += 1; t += d
    (OUT / "captions.srt").write_text("\n".join(out), encoding="utf-8")


def mix_and_encode(marks, total):
    ins, filt = [], []
    for i, (key, start) in enumerate(marks):
        ins += ["-i", str(OUT / ("scene_%s.wav" % key))]
        filt.append("[%d:a]adelay=%d|%d[a%d]" % (i, int(start * 1000), int(start * 1000), i))
    filt.append("%samix=inputs=%d:normalize=0:duration=longest[mix]" % ("".join("[a%d]" % i for i in range(len(marks))), len(marks)))
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", *ins, "-filter_complex", ";".join(filt),
                    "-map", "[mix]", "-ar", "44100", str(OUT / "narration.wav")], check=True)
    style = ("FontName=Arial,FontSize=15,PrimaryColour=&H00FFFFFF,BackColour=&HA0000000,"
             "BorderStyle=4,Outline=0,Shadow=0,MarginV=10,MarginL=40,MarginR=40")
    # captions sit in a dark strip BELOW the picture, so they never cover the app
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", "raw.webm", "-i", "narration.wav",
                    "-vf", "pad=%d:%d:0:0:0x111111,subtitles=captions.srt:force_style='%s'" % (W, H + CAP_H, style),
                    "-c:v", "libx264", "-preset", "medium", "-crf", "24", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-b:a", "160k", "-af", "loudnorm=I=-16:TP=-1.5:LRA=11", "-ar", "48000",
                    "-t", "%.2f" % total, "-movflags", "+faststart", "Reliance_Under_Disruption_demo.mp4"],
                   check=True, cwd=str(OUT))


def main():
    print("1/4 synthesising narration (Microsoft Zira, offline)")
    durs = synthesise()
    print("    %d scenes, %.0f s of narration" % (len(durs), sum(durs.values())))
    print("2/4 recording the live session")
    marks, total, verify_text = record(durs)
    print("    recorded %.0f s" % total)
    print("3/4 captions")
    captions(marks, durs)
    print("4/4 mixing narration and encoding")
    mix_and_encode(marks, total)
    tr = ["# Narrated demo: transcript", ""]
    for key, start in marks:
        tr.append("**[%s] %s**  %s\n" % (srt_time(start)[:8], key, NARR[key]))
    tr += ["## Verification output of the recorded session", "", "```", verify_text.strip(), "```"]
    (OUT / "transcript.md").write_text("\n".join(tr), encoding="utf-8")
    print("done ->", OUT / "Reliance_Under_Disruption_demo.mp4")


if __name__ == "__main__":
    main()
