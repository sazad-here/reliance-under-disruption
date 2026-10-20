"""Figure 1, authored as SVG and rasterised through the local Chromium.

Why SVG rather than the AI image skills: every image-generation skill in the
vault needs a credential (OpenRouter, Gemini, fal, Stability) and none is
present. They would also have sent an unpublished manuscript to a third party
and, by their own documentation, routinely misspell labels -- unacceptable for
a figure carrying "194 trees" and "AUROC 0.897".

Why SVG rather than matplotlib: precise typographic control, real vector
output, and room for a third panel without the layout fighting back.

Redesigned 19 Oct 2026:
  * The canvas is 820 units for the 122 mm print width (it was 1220), so the
    same type prints at about 5.5-7 pt instead of 3.7-4.3 pt.
  * Colour now carries meaning: warm = what the person does, blue = what the
    AI contributes. Every coloured element also has an icon or a label, so the
    figure still reads in greyscale.
  * Panel (c) shows the study's manipulation itself: the weeks eligible for a
    propagating or an absorbed scripted error, read from
    results/placement_design.json, against the six disruption weeks. Week
    numbers are the twin's period index, the same scale as Fig. 3.

Run:  python figures/make_figure1_svg.py
Out:  figures/paper/main_text/Figure1_merged_workflow.png (+ .svg)
"""
import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "paper" / "main_text" / "Figure1_merged_workflow.png"
SVG = HERE / "paper" / "main_text" / "Figure1_merged_workflow.svg"

W, H = 820, 304           # SVG user units; 820 maps to the 122 mm print width
INK = "#1a1a19"
MUTED = "#55534d"
HUMAN_F, HUMAN_S = "#fcebdd", "#a6500f"   # the person's steps (warm)
AI_F, AI_S = "#e3eef8", "#0b5e99"         # the AI's contribution (blue)
FROZEN = "#0b5e99"                         # the frozen model, solid
PROP = "#d55e00"                           # propagating-error week (Okabe-Ito vermillion)
ABS = "#0072b2"                            # absorbed-error week (Okabe-Ito blue), hatched
PRACTICE = "#e6e4dd"

# Arial, the same face as Figs. 2 and 3, so the three figures read as one set
FONT = "Arial, Helvetica, sans-serif"

# Simple stroked glyphs, 24x24 in their own coordinate space.
ICON = {
    "data": '<ellipse cx="12" cy="6" rx="8" ry="3.1"/>'
            '<path d="M4 6 V12 a8 3.1 0 0 0 16 0 V6"/>'
            '<path d="M4 12 V17 a8 3.1 0 0 0 16 0 V12"/>',
    "twin": '<path d="M3 20 V10 L12 4 L21 10 V20 Z"/><path d="M9 20 V14 H15 V20"/>',
    "grid": '<rect x="3" y="3" width="18" height="18" rx="1.5"/>'
            '<path d="M9 3 V21 M15 3 V21 M3 9 H21 M3 15 H21"/>',
    "tree": '<path d="M12 21 V13"/><path d="M12 13 L6 8 M12 13 L18 8"/>'
            '<path d="M6 8 L3 4 M6 8 L9 4 M18 8 L15 4 M18 8 L21 4"/>',
    "lock": '<rect x="4.5" y="10" width="15" height="10" rx="2"/>'
            '<path d="M8 10 V7 a4 4 0 0 1 8 0 V10"/>',
    "eye": '<path d="M2 12 C6 5.5, 18 5.5, 22 12 C18 18.5, 6 18.5, 2 12 Z"/>'
           '<circle cx="12" cy="12" r="3.1"/>',
    "pen": '<path d="M4 20 L5.4 15.2 L16.6 4 L20 7.4 L8.8 18.6 Z"/><path d="M14.6 6 L18 9.4"/>',
    "chip": '<rect x="6" y="6" width="12" height="12" rx="1.6"/>'
            '<path d="M10 2 V6 M14 2 V6 M10 18 V22 M14 18 V22'
            ' M2 10 H6 M2 14 H6 M18 10 H22 M18 14 H22"/>',
    "tick": '<path d="M3.5 12.8 L9 18.2 L20.5 6.4"/>',
}


def icon(name, x, y, size=14, color=INK):
    sc = size / 24.0
    return ('<g transform="translate(%g,%g) scale(%g)" fill="none" stroke="%s" '
            'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">%s</g>'
            % (x, y, sc, color, ICON[name]))


def text(x, y, t, size=12, anchor="middle", weight="400", fill=INK, italic=False):
    st = ' font-style="italic"' if italic else ""
    return ('<text x="%g" y="%g" text-anchor="%s" font-family="%s" font-size="%g" '
            'font-weight="%s" fill="%s"%s>%s</text>' % (x, y, anchor, FONT, size, weight, fill, st, t))


def box(x, y, w, h, title, sub, fill, stroke, ic, tcolor=INK):
    s = ['<rect x="%g" y="%g" width="%g" height="%g" rx="6" fill="%s" stroke="%s" stroke-width="1.3"/>'
         % (x, y, w, h, fill, stroke)]
    tw = len(title) * 15.5 * 0.55
    s.append(icon(ic, x + w / 2 - tw / 2 - 18, y + 7, 14, tcolor))
    s.append(text(x + w / 2 - tw / 2, y + 19, title, 15.5, "start", "650", tcolor))
    for i, ln in enumerate(sub):
        s.append(text(x + w / 2, y + 35.5 + i * 14, ln, 13, fill=tcolor))
    return "".join(s)


def arrow(x1, y1, x2, y2, color=INK, dashed=False, w=1.4):
    d = ' stroke-dasharray="4.5,3"' if dashed else ""
    return ('<line x1="%g" y1="%g" x2="%g" y2="%g" stroke="%s" stroke-width="%g"%s '
            'marker-end="url(#ah)"/>' % (x1, y1, x2, y2, color, w, d))


def panel_title(y, letter, t):
    return text(0, y, letter, 16, "start", "700") + text(17, y, t, 14.5, "start", "600")


def build():
    pd = json.loads((ROOT / "results" / "placement_design.json").read_text(encoding="utf-8"))
    prop, absb = set(pd["propagating"]), set(pd["absorbed"])
    disruptions = (6, 9, 13, 17, 21, 26)     # the fixed calendar, src/simulator.py

    # 2-unit margin all round, so outer box strokes are not clipped at the canvas edge
    p = ['<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" viewBox="-2 -2 %d %d">' % (W + 4, H + 4, W + 4, H + 4),
         '<defs>'
         '<marker id="ah" markerWidth="7" markerHeight="7" refX="6" refY="2.6" orient="auto">'
         '<path d="M0,0 L0,5.2 L6.2,2.6 z" fill="%s"/></marker>'
         '<marker id="ahb" markerWidth="7" markerHeight="7" refX="6" refY="2.6" orient="auto">'
         '<path d="M0,0 L0,5.2 L6.2,2.6 z" fill="%s"/></marker>'
         '<pattern id="hatch" width="4" height="4" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">'
         '<rect width="4" height="4" fill="white"/><line x1="0" y1="0" x2="0" y2="4" stroke="%s" stroke-width="1.6"/>'
         '</pattern></defs>' % (INK, AI_S, ABS),
         '<rect x="-2" y="-2" width="%d" height="%d" fill="white"/>' % (W + 4, H + 4)]

    # ------------------------------------------------------------- (a) model development
    p.append(panel_title(13, "a", "Building the risk model, once, before any participant"))
    stages = [("Open data", ["M5, Olist,", "DataCo"], "data"),
              ("Simulation", ["520 episodes", "× 100 weeks"], "twin"),
              ("Features", ["30 inputs: state", "+ 4-week history"], "grid"),
              ("LightGBM", ["194 trees,", "Platt-calibrated"], "tree")]
    bw, bh, gap, ya = 136, 58, 35, 22
    for i, (t, sub, ic) in enumerate(stages):
        x = i * (bw + gap)
        p.append(box(x, ya, bw, bh, t, sub, AI_F, AI_S, ic))
        if i:
            p.append(arrow(x - gap + 3, ya + bh / 2, x - 3, ya + bh / 2, AI_S))
    mx = 4 * (bw + gap)
    p.append(box(mx, ya, W - mx, bh, "Frozen model", ["AUROC 0.897", "ECE 0.006"], FROZEN, FROZEN, "lock", "white"))
    p.append(arrow(mx - gap + 3, ya + bh / 2, mx - 3, ya + bh / 2, AI_S))

    # ------------------------------------------------------------- (b) one decision week
    yb = 111
    p.append(panel_title(yb, "b", "One decision week: the participant stays in charge"))
    steps = [("1 Inspect", ["stock, demand,", "arrivals, backorders"], "eye", "h"),
             ("2 Own order", ["committed and", "locked first"], "pen", "h"),
             # the order is scripted (Sect. 3.5); only the risk % comes from the frozen model
             ("3 AI advice", ["scripted order", "+ model’s live risk %"], "chip", "a"),
             ("4 Final order", ["accept, modify", "or override"], "tick", "h")]
    sw, sh, sg, ys = 176, 58, 38, yb + 12
    xs = []
    for i, (t, sub, ic, who) in enumerate(steps):
        x = i * (sw + sg)
        xs.append(x)
        f, s = (HUMAN_F, HUMAN_S) if who == "h" else (AI_F, AI_S)
        p.append(box(x, ys, sw, sh, t, sub, f, s, ic))
        if i:
            p.append(arrow(x - sg + 3, ys + sh / 2, x - 3, ys + sh / 2))
    # the lock between own order and advice: this is what makes weight of advice measurable
    lx = xs[2] - sg / 2
    p.append('<line x1="%g" y1="%g" x2="%g" y2="%g" stroke="%s" stroke-width="1.8" stroke-dasharray="2.5,2.5"/>'
             % (lx, ys - 2, lx, ys + sh + 2, HUMAN_S))
    p.append(icon("lock", lx - 6, ys + sh + 4, 12, HUMAN_S))
    # the frozen model feeds the advice step
    cx = mx + (W - mx) / 2
    p.append('<path d="M %g %g C %g %g, %g %g, %g %g" fill="none" stroke="%s" stroke-width="1.4" '
             'stroke-dasharray="4.5,3" marker-end="url(#ahb)"/>'
             % (cx, ya + bh + 2, cx, ya + bh + 26, xs[2] + sw * 0.62, ys - 30, xs[2] + sw * 0.62, ys - 3, AI_S))
    # on the curve's horizontal run, between the curve and the boxes above, so it reads as
    # the curve's label (Arial runs wider than the old face, which clipped it at the edge)
    p.append(text((cx + xs[2] + sw * 0.62) / 2 + 20, ya + bh + 11, "live risk %", 12.5, "middle", "600", AI_S))
    # state carries into the next week
    yl = ys + sh + 22
    p.append('<path d="M %g %g L %g %g L %g %g L %g %g" fill="none" stroke="%s" stroke-width="1.5" marker-end="url(#ah)"/>'
             % (xs[3] + sw / 2, ys + sh + 2, xs[3] + sw / 2, yl, xs[0] + sw / 2, yl, xs[0] + sw / 2, ys + sh + 4, INK))
    p.append('<rect x="%g" y="%g" width="%g" height="16" fill="white"/>' % (W / 2 - 188, yl - 9, 376))
    p.append(text(W / 2, yl + 4, "inventory, pipeline orders and backorders carry into next week", 12.2, fill=MUTED, italic=True))

    # ------------------------------------------------------------- (c) one session
    yc = 238
    p.append(panel_title(yc, "c", "One session: where disruptions and scripted errors fall"))
    cells, cw = 35, W / 35.0
    ty, th = yc + 20, 17
    # five practice weeks as one block, so its label has room
    p.append('<rect x="0.6" y="%g" width="%g" height="%g" fill="%s" stroke="%s" stroke-width="0.7"/>'
             % (ty, 5 * cw - 1.2, th, PRACTICE, MUTED))
    p.append(text(5 * cw / 2, ty + 12.5, "5 practice weeks", 11.5, fill=INK))
    for i in range(5, cells):
        x = i * cw
        wk = i - 5
        fill = PROP if wk in prop else ("url(#hatch)" if wk in absb else "white")
        p.append('<rect x="%g" y="%g" width="%g" height="%g" fill="%s" stroke="%s" stroke-width="0.7"/>'
                 % (x + 0.6, ty, cw - 1.2, th, fill, MUTED))
    for wk in disruptions:
        x = (5 + wk) * cw + cw / 2
        p.append('<path d="M %g %g L %g %g L %g %g Z" fill="%s"/>' % (x - 4.2, ty - 9, x + 4.2, ty - 9, x, ty - 2, INK))
    # legend
    ly = ty + th + 15
    lg = []
    lx0 = 5 * cw
    lg.append('<rect x="%g" y="%g" width="13" height="10" fill="%s" stroke="%s" stroke-width="0.7"/>' % (lx0, ly - 9, PROP, MUTED))
    lg.append(text(lx0 + 18, ly, "error that propagates", 12.2, "start"))
    lx1 = lx0 + 168
    lg.append('<rect x="%g" y="%g" width="13" height="10" fill="url(#hatch)" stroke="%s" stroke-width="0.7"/>' % (lx1, ly - 9, MUTED))
    lg.append(text(lx1 + 18, ly, "error that is absorbed", 12.2, "start"))
    lx2 = lx1 + 168
    lg.append('<path d="M %g %g L %g %g L %g %g Z" fill="%s"/>' % (lx2, ly - 9, lx2 + 9, ly - 9, lx2 + 4.5, ly - 1, INK))
    lg.append(text(lx2 + 15, ly, "disruption", 12.2, "start"))
    lx3 = lx2 + 92
    lg.append(text(lx3, ly, "(candidate weeks; 30 main weeks)", 11, "start", fill=MUTED, italic=True))
    p.extend(lg)

    p.append("</svg>")
    return "".join(p)


def main():
    SVG.parent.mkdir(parents=True, exist_ok=True)
    SVG.write_text(build(), encoding="utf-8")
    print("wrote", SVG.name)

    from playwright.sync_api import sync_playwright
    scale = 6  # 824 px * 6 = 4944 px across 122 mm -> about 1030 dpi
    with sync_playwright() as pw:
        b = pw.chromium.launch()
        pg = b.new_page(viewport={"width": W + 4, "height": H + 4}, device_scale_factor=scale)
        pg.goto(SVG.as_uri())
        pg.wait_for_timeout(350)
        pg.screenshot(path=str(OUT), omit_background=False)
        b.close()
    print("wrote", OUT.name)
    from PIL import Image
    im = Image.open(OUT)
    print("%dx%d px -> at 122 mm = %.1f mm tall; 12-unit text prints at %.1f pt"
          % (im.size[0], im.size[1], 122 * im.size[1] / im.size[0], 12 * 122 / W / 0.3528))


if __name__ == "__main__":
    main()
