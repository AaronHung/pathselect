#!/usr/bin/env python3
"""Fig.1 v3.0 —— 純程式生成（規格：Fig1_v3_最終畫圖規格_Cursor.md，2026-09-10）。

    python scripts/make_fig1.py            # 產出 svg / pdf / png 並跑守門
    python scripts/make_fig1.py --check    # 只對既有產物跑守門

輸出 paper/figures/fig1_architecture_v1_0.{svg,pdf,png}。

設計決定（與規格的對應）：
* 畫布 504 × 214 pt = 7.0 in × 2.97 in（\\textwidth 下高度 ≤ 3.0 in，比約 2.36:1）。
  單位就是 pt，所以檔內字級即為排版後字級；**全部 ≥ 8.5 pt**。
* ❄ 畫成 path 而非文字 —— 舊版靠 Apple Color Emoji，cairo 會把點陣字型嵌成 Type 3，
  違反「pdffonts 無 Type 3」。畫成 path 就沒有字型問題。
* **PDF 用 Chrome headless 轉，不用 rsvg-convert**：實測 librsvg/cairo 會把**空格**
  （無輪廓的字形）嵌成一個空的 Type 3 字型，任何含空格的文字都觸發、換字型無效
  （Helvetica Neue／Helvetica／Arial 皆然）。Chrome 以 CID TrueType 嵌入、頁面
  尺寸由 `@page` 精確控制。PNG 仍用 rsvg-convert（點陣圖無字型嵌入問題）。
* 下標依規格字面寫 `U_old` / `b_j` / `L_KD`（規格「精確文字」，守門查的就是這些字串）。
* 唯一橘色 = Hierarchical Selector；其餘盒子灰（帶 ❄）或白。
* 縮圖一律中性色塊 —— PI 未提供真實 TCGA 裁片，規格禁止合成仿病理圖。
* 副標（規格「可選」）未放：頂列已有兩個 callout，再放會擠壓字級。
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "paper" / "figures"
STEM = "fig1_architecture_v1_0"

W, H = 504.0, 214.0                  # pt；7.0 in × 2.97 in
FONT = "Helvetica Neue, Helvetica, Arial, sans-serif"
FS = 8.5                             # 最小字級（規格下限）
FS_TITLE = 9.0
LH = 10.0                            # 行距

# ── 顏色 ────────────────────────────────────────────────────────────────────
ORANGE_FILL, ORANGE_STROKE = "#FFE3C8", "#E8772E"      # 只准 selector 用
GRAY_FILL, GRAY_STROKE = "#EDEDED", "#8C8C8C"
WHITE_FILL, DARK_STROKE = "#FFFFFF", "#4A4A4A"
CALLOUT_FILL, CALLOUT_STROKE = "#FFF6D6", "#C9A227"
RED = "#C0392B"
ICE = "#3E86C9"                                          # ❄
TEXT = "#1A1A1A"
#: 8 個組織群組的顏色（不含橘色，避免與 selector 混淆；色盲友善）
GROUP_COLORS = ["#0072B2", "#009E73", "#56B4E9", "#CC79A7",
                "#E6C700", "#8E44AD", "#1B9E77", "#8C6D31"]
#: 示意用配額：8 個證據方塊分屬哪些群組（Σ = 8）。僅示意，不對應任何實驗。
EVIDENCE_QUOTA = [3, 2, 1, 1, 1, 0, 0, 0]

# ── 守門清單（規格第四節） ───────────────────────────────────────────────────
REQUIRED = ["Only the selector learns", "Preserve usefulness", "measured directly",
            "no images, no features", "U_old"]
FORBIDDEN = ["Equivalence", "Segmentation", "CURRENT DIAGNOSIS", "U_oal",
             "512", "3.1 MB", "256", "Task 1", "Task 2", "Task T",
             "KL(", "max(0", "‖", "∥"]


# ── SVG 小工具 ───────────────────────────────────────────────────────────────

def esc(s: str) -> str:
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
             .replace('"', "&quot;"))


def text(x, y, s, *, size=FS, weight="normal", anchor="middle", fill=TEXT,
         cls="") -> str:
    c = f' class="{cls}"' if cls else ""
    return (f'<text x="{x:.2f}" y="{y:.2f}" font-family="{FONT}" font-size="{size}" '
            f'font-weight="{weight}" text-anchor="{anchor}" fill="{fill}"{c}>{esc(s)}</text>')


def lines(x, y_top, rows, *, size=FS, anchor="middle", fill=TEXT, lh=LH,
          bold_first=False) -> str:
    """多行文字，y_top 是第一行的基線。"""
    out = []
    for i, s in enumerate(rows):
        f = fill[i] if isinstance(fill, (list, tuple)) else fill
        out.append(text(x, y_top + i * lh, s, size=size, anchor=anchor, fill=f,
                        weight="bold" if (bold_first and i == 0) else "normal"))
    return "\n".join(out)


def rect(x, y, w, h, *, fill, stroke, sw=0.8, r=3.0, dash=None, cls="") -> str:
    d = f' stroke-dasharray="{dash}"' if dash else ""
    c = f' class="{cls}"' if cls else ""
    return (f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" rx="{r}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}{c}/>')


def snowflake(cx, cy, r=3.2) -> str:
    """❄ 畫成 path：三條主軸 + 每端一對短叉。"""
    import math
    p = []
    for k in range(3):
        a = math.radians(60 * k)
        dx, dy = r * math.cos(a), r * math.sin(a)
        p.append(f"M{cx-dx:.2f},{cy-dy:.2f} L{cx+dx:.2f},{cy+dy:.2f}")
        for sgn in (1, -1):                       # 兩端的短叉
            ex, ey = cx + sgn * dx * 0.62, cy + sgn * dy * 0.62
            for t in (a + math.radians(55), a - math.radians(55)):
                tx, ty = ex + sgn * 1.3 * math.cos(t), ey + sgn * 1.3 * math.sin(t)
                p.append(f"M{ex:.2f},{ey:.2f} L{tx:.2f},{ty:.2f}")
    return (f'<path d="{" ".join(p)}" stroke="{ICE}" stroke-width="0.8" '
            f'stroke-linecap="round" fill="none" class="snowflake"/>')


def arrow(x1, y1, x2, y2, *, dashed=False, sw=0.9, cls="", color=DARK_STROKE) -> str:
    d = ' stroke-dasharray="2.4,1.8"' if dashed else ""
    c = f' class="{cls}"' if cls else ""
    return (f'<path d="M{x1:.2f},{y1:.2f} L{x2:.2f},{y2:.2f}" stroke="{color}" '
            f'stroke-width="{sw}" fill="none"{d} marker-end="url(#ah)"{c}/>')


# ── 版面 ────────────────────────────────────────────────────────────────────
# 盒寬依 Helvetica Neue 實測字寬（PIL getlength）＋ ≥4 pt 內距：
#   Frozen CONCH(bold 9) 65.6 → 70；8 fixed / prompts 換行 → 40；Whole- / slide / image → 40；
#   Hierarchical Selector (learned)(bold 9) 129.9 → 152（含三個子盒 36/44/44 + 兩個 → 間距 10，內距 4）；
#   |P| = B = 8 39.6、8 tiles 52.4 → 56；pool selected 51.2 → 56；Diagnosis(bold 9) 42.5 → 48。
# 頂列 x 區間（左邊界, 寬）—— 7 個盒子，邊距 4、間距 5：40+70+44+152+56+56+48 + 6×5 = 496
BOX = {"wsi": (4, 40), "conch": (49, 70), "tissue": (124, 44), "sel": (173, 152),
       "evid": (330, 56), "head": (391, 56), "diag": (452, 48)}
TOP_Y, TOP_H = 28.0, 58.0            # 一般盒子
SEL_Y, SEL_H = 24.0, 66.0            # selector 較高
DIV_Y = 99.0                         # 上下分隔線
TITLE_Y = (108.0, 117.0)             # 訓練帶標題兩行的基線（單行會被 L_KD 虛線箭頭穿過）
BAND_Y, BAND_H = 120.0, 84.0         # 訓練帶外框
LBL_Y, LBL_H = 131.0, 50.0           # 三個標籤盒（competence) 49.7 → 58）
LBL = {"kd": (104, 58), "eq": (170, 58), "rep": (236, 58)}
MEM = (4, 92)                        # 記憶庫盒 (x, w)；no images, no features 87.0 → 92
LORA, SHARED = (312, 96), (426, 74)
LEGEND_Y = 211.0
LEGEND_TEXT = ("frozen (no training) · orange = learned · dashed = training only · "
               "solid path = training and inference")
LEGEND_W = 371.0                     # 實測 370.6 pt（8.5 pt Helvetica Neue）


def build() -> str:
    P = []  # parts
    P.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}pt" height="{H}pt" '
             f'viewBox="0 0 {W} {H}">')
    P.append('<defs><marker id="ah" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" '
             'markerHeight="6" orient="auto-start-reverse">'
             f'<path d="M0,0 L8,4 L0,8 z" fill="{DARK_STROKE}"/></marker></defs>')
    P.append(f'<rect x="0" y="0" width="{W}" height="{H}" fill="#FFFFFF"/>')

    def cx(key, table=BOX):
        x, w = table[key]; return x + w / 2

    # ── 頂列七盒 ────────────────────────────────────────────────────────────
    # 1 Whole-slide image：灰色方塊 + 格線（中性，不合成病理圖）
    x, w = BOX["wsi"]
    P.append(rect(x, TOP_Y, w, TOP_H, fill=WHITE_FILL, stroke=DARK_STROKE))
    P.append(lines(cx("wsi"), TOP_Y + 11, ["Whole-", "slide", "image"], bold_first=True))
    gx, gy, gw, gh = x + 8, TOP_Y + 36, w - 16, 18
    P.append(rect(gx, gy, gw, gh, fill="#D6D6D6", stroke="#9A9A9A", sw=0.5, r=1))
    for i in range(1, 3):
        P.append(f'<line x1="{gx+gw*i/3:.2f}" y1="{gy}" x2="{gx+gw*i/3:.2f}" y2="{gy+gh}" '
                 f'stroke="#FFFFFF" stroke-width="0.6"/>')
        P.append(f'<line x1="{gx}" y1="{gy+gh*i/3:.2f}" x2="{gx+gw}" y2="{gy+gh*i/3:.2f}" '
                 f'stroke="#FFFFFF" stroke-width="0.6"/>')

    # 2 Frozen CONCH encoder ❄ / patch features {x_i}   （❄ 緊接 encoder 之後）
    x, w = BOX["conch"]
    P.append(rect(x, TOP_Y, w, TOP_H, fill=GRAY_FILL, stroke=GRAY_STROKE))
    P.append(lines(cx("conch"), TOP_Y + 13, ["Frozen CONCH", "encoder", "patch features", "{x_i}"],
                   bold_first=True))
    P.append(snowflake(cx("conch") + 23, TOP_Y + 20))

    # 3 Tissue grouping / 8 fixed prompts ❄ —— 灰、8 色小方塊（❄ 緊接 prompts 之後）
    x, w = BOX["tissue"]
    P.append(rect(x, TOP_Y, w, TOP_H, fill=GRAY_FILL, stroke=GRAY_STROKE))
    P.append(lines(cx("tissue") - 2.5, TOP_Y + 10, ["Tissue", "grouping", "8 fixed", "prompts"],
                   bold_first=True, lh=9.6))
    P.append(snowflake(cx("tissue") + 18.5, TOP_Y + 35.8, r=2.9))
    s, g = 5.4, 1.2
    ox = cx("tissue") - (4 * s + 3 * g) / 2
    for i, c in enumerate(GROUP_COLORS):
        r_, c_ = divmod(i, 4)
        P.append(rect(ox + c_ * (s + g), TOP_Y + 43.5 + r_ * (s + g), s, s,
                      fill=c, stroke="none", r=1, cls="group-swatch"))

    # 4 Hierarchical Selector (learned) —— 唯一橘色、最大、粗框
    x, w = BOX["sel"]
    P.append(rect(x, SEL_Y, w, SEL_H, fill=ORANGE_FILL, stroke=ORANGE_STROKE, sw=1.8,
                  r=4, cls="selector"))
    P.append(text(cx("sel"), SEL_Y + 11, "Hierarchical Selector (learned)", size=FS_TITLE,
                  weight="bold"))
    steps = [(36.0, ["Group", "scoring"]), (44.0, ["Budget", "allocation"]),
             (44.0, ["Patch", "scoring", "+ top-b_j"])]
    sg, pad, st, sh = 10.0, 4.0, SEL_Y + 16, 34.0          # 子盒間距／內距／頂／高
    bx = x + pad
    for i, (bw, rows) in enumerate(steps):
        P.append(rect(bx, st, bw, sh, fill=WHITE_FILL, stroke=ORANGE_STROKE, sw=0.7, r=2))
        y0 = st + (sh - (len(rows) - 1) * LH) / 2 + 3.0      # 垂直置中
        P.append(lines(bx + bw / 2, y0, rows))
        if i < 2:
            P.append(text(bx + bw + sg / 2, st + sh / 2 + 3, "→", size=FS))
        bx += bw + sg
    P.append(text(cx("sel"), SEL_Y + 60, "Group → Patch · Σ_j b_j = B = 8", size=FS))

    # 5 Selected evidence —— 8 方塊依配額上色；盒下標籤（起點對齊盒左緣，避開 selector 底邊）
    x, w = BOX["evid"]
    P.append(rect(x, TOP_Y, w, TOP_H, fill=WHITE_FILL, stroke=DARK_STROKE))
    P.append(lines(cx("evid"), TOP_Y + 11, ["Selected", "evidence", "|P| = B = 8"],
                   bold_first=True))
    tiles = [c for c, n in zip(GROUP_COLORS, EVIDENCE_QUOTA) for _ in range(n)]
    assert len(tiles) == 8
    ts, tg = 5.5, 1.2
    tx0 = cx("evid") - (8 * ts + 7 * tg) / 2
    for i, c in enumerate(tiles):
        P.append(rect(tx0 + i * (ts + tg), TOP_Y + 42, ts, ts, fill=c, stroke="none",
                      r=1, cls="evidence-tile"))
    P.append(text(x + 1, TOP_Y + TOP_H + 8.5, "hard budgeted evidence", size=FS, anchor="start"))

    # 6 Frozen diagnosis ❄（❄ 緊接 diagnosis 之後）
    x, w = BOX["head"]
    P.append(rect(x, TOP_Y, w, TOP_H, fill=GRAY_FILL, stroke=GRAY_STROKE))
    P.append(lines(cx("head"), TOP_Y + 11, ["Frozen", "diagnosis", "pool selected", "evidence",
                                             "cosine with", "fixed {t_c}"],
                   bold_first=True, lh=8.6))
    P.append(snowflake(cx("head") + 24, TOP_Y + 16.8))

    # 7 Diagnosis p(y | P) / inference ends here
    x, w = BOX["diag"]
    P.append(rect(x, TOP_Y, w, TOP_H, fill=WHITE_FILL, stroke=DARK_STROKE))
    P.append(lines(cx("diag"), TOP_Y + 15, ["Diagnosis", "p(y | P)"], bold_first=True))
    P.append(lines(cx("diag"), TOP_Y + 42, ["inference", "ends here"], fill="#555555"))

    # 頂列實線箭頭（訓練與推論共用路徑）
    ym = TOP_Y + TOP_H / 2
    order = ["wsi", "conch", "tissue", "sel", "evid", "head", "diag"]
    for a, b in zip(order, order[1:]):
        x1 = BOX[a][0] + BOX[a][1]; x2 = BOX[b][0]
        P.append(arrow(x1 + 0.5, ym, x2 - 0.5, ym))

    # ── callout ① ③ ───────────────────────────────────────────────────────
    P.append(rect(52, 4, 110, 15, fill=CALLOUT_FILL, stroke=CALLOUT_STROKE, sw=0.7, r=7))
    P.append(text(107, 14.5, "Only the selector learns", weight="bold"))
    P.append(f'<path d="M150,19 L{BOX["sel"][0]+2},{SEL_Y+2}" stroke="{CALLOUT_STROKE}" '
             'stroke-width="0.7"/>')
    ex, ew = BOX["evid"]
    P.append(rect(ex, 2, 150, 23, fill=CALLOUT_FILL, stroke=CALLOUT_STROKE, sw=0.7, r=7))
    P.append(lines(ex + 75, 11, ["Selection drift measured directly",
                                 "identity (Jaccard) · utility (ΔU)"], bold_first=True, lh=9.5))
    P.append(f'<path d="M{ex+ew/2},25 L{ex+ew/2},{TOP_Y}" stroke="{CALLOUT_STROKE}" '
             'stroke-width="0.7"/>')

    # ── 分隔線與訓練帶 ─────────────────────────────────────────────────────
    P.append(f'<line x1="4" y1="{DIV_Y}" x2="{W-4}" y2="{DIV_Y}" stroke="#B0B0B0" '
             'stroke-width="0.6" stroke-dasharray="1.5,2"/>')
    P.append(text(4, TITLE_Y[0], "Continual preservation during training", anchor="start",
                  weight="bold"))
    P.append(text(4, TITLE_Y[1], "(training only)", anchor="start", weight="bold"))
    P.append(rect(4, BAND_Y, W - 8, BAND_H, fill="none", stroke="#B0B0B0", sw=0.6, r=4,
                  dash="3,2"))

    # 8 Selection Memory
    x, w = MEM
    my, mh = BAND_Y + 4, BAND_H - 8
    P.append(rect(x, my, w, mh, fill=WHITE_FILL, stroke=DARK_STROKE))
    P.append(lines(x + w / 2, my + 12,
                   ["Selection Memory", "past selector", "snapshots", "scores · candidate",
                    "indices · utilities", "no images, no features"],
                   bold_first=True, lh=9.6,
                   fill=[TEXT, TEXT, TEXT, TEXT, TEXT, RED]))

    # 9 三個標籤盒（虛線框）；L_eq 最粗
    labels = {"kd": (["Distill old", "scores", "(behavioral", "continuity)", "L_KD"], 0.9),
              "eq": (["Preserve", "utility", "(functional", "continuity)", "L_eq"], 1.9),
              "rep": (["Replay old", "samples", "(task", "competence)", "L_rep"], 0.9)}
    for k, (rows, swid) in labels.items():
        lx, lw = LBL[k]
        P.append(rect(lx, LBL_Y, lw, LBL_H, fill=WHITE_FILL, stroke=DARK_STROKE,
                      sw=0.7, dash="2.4,1.8"))
        P.append(lines(lx + lw / 2, LBL_Y + 10, rows, bold_first=True, lh=9.4))
    # 記憶庫 ┈▶ 標籤列
    P.append(arrow(x + w + 0.5, LBL_Y + LBL_H / 2, LBL["kd"][0] - 0.5, LBL_Y + LBL_H / 2,
                   dashed=True))
    # 三條虛線箭頭：標籤盒頂 → selector 底（終點都在 selector 內）
    sel_x, sel_w = BOX["sel"]; sel_bottom = SEL_Y + SEL_H
    targets = {"kd": sel_x + 12, "eq": sel_x + sel_w / 2, "rep": sel_x + sel_w - 12}
    for k, (rows, swid) in labels.items():
        lx, lw = LBL[k]
        P.append(arrow(lx + lw / 2, LBL_Y - 0.5, targets[k], sel_bottom + 0.5,
                       dashed=True, sw=swid, cls="to-selector"))

    # callout ②（貼在 Preserve utility 旁：置中於 L_eq 盒之下；文字實測 157.4 → 寬 170）
    eqc = LBL["eq"][0] + LBL["eq"][1] / 2
    P.append(rect(eqc - 85, LBL_Y + LBL_H + 2, 170, 19, fill=CALLOUT_FILL,
                  stroke=CALLOUT_STROKE, sw=0.7, r=7))
    P.append(lines(eqc, LBL_Y + LBL_H + 9.5,
                   ["Preserve usefulness, not exact identity",
                    "allow different evidence if U_new ≥ U_old"], bold_first=True, lh=8.6))

    # 10 Task-local LoRA update → One shared selector
    lx, lw = LORA
    P.append(rect(lx, LBL_Y, lw, LBL_H, fill=WHITE_FILL, stroke=DARK_STROKE))
    P.append(lines(lx + lw / 2, LBL_Y + 10,
                   ["Task-local", "LoRA update", "fresh ΔW_τ ·", "fold into", "shared selector"],
                   bold_first=True, lh=9.4))
    sx, sw2 = SHARED
    P.append(rect(sx, LBL_Y, sw2, LBL_H, fill=WHITE_FILL, stroke=DARK_STROKE))
    P.append(lines(sx + sw2 / 2, LBL_Y + 14, ["One shared", "selector", "(used at", "test time)"],
                   bold_first=True, lh=9.4))
    P.append(arrow(LBL["rep"][0] + LBL["rep"][1] + 0.5, LBL_Y + LBL_H / 2, lx - 0.5,
                   LBL_Y + LBL_H / 2, dashed=True))
    P.append(arrow(lx + lw + 0.5, LBL_Y + LBL_H / 2, sx - 0.5, LBL_Y + LBL_H / 2))

    # ── 圖例（右下一行）───────────────────────────────────────────────────
    lx0 = W - 4 - LEGEND_W                       # 文字左緣（靠右對齊後回推）
    P.append(snowflake(lx0 - 6, LEGEND_Y - 3, r=2.8))
    P.append(text(lx0, LEGEND_Y, LEGEND_TEXT, anchor="start", fill="#444444"))
    P.append("</svg>")
    return "\n".join(P) + "\n"


# ── PDF：Chrome headless ─────────────────────────────────────────────────────

CHROME_CANDIDATES = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
]


def svg_to_pdf_chrome(svg: Path, pdf: Path, work: Path) -> None:
    chrome = next((c for c in CHROME_CANDIDATES if Path(c).exists()), None)
    if chrome is None:
        raise SystemExit("❌ 找不到 Chrome/Chromium/Brave —— PDF 需要它才能避免 cairo 的 "
                         "空格 Type 3 問題（見模組說明）")
    html = work / "fig1.html"
    html.write_text(
        '<!doctype html><html><head><meta charset="utf-8"><style>'
        f'@page{{size:{W}pt {H}pt;margin:0}} html,body{{margin:0;padding:0}} '
        'svg{display:block}</style></head><body>' + svg.read_text() + "</body></html>")
    subprocess.run([chrome, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={pdf}", f"file://{html}"],
                   check=True, capture_output=True)
    if not pdf.exists() or pdf.stat().st_size == 0:
        raise SystemExit("❌ Chrome 沒有產出 PDF")


# ── 守門 ────────────────────────────────────────────────────────────────────

def gate(svg_path: Path, pdf_path: Path) -> list[str]:
    """回傳違規清單；空 = 通過。每條都印出來。"""
    bad = []
    raw = svg_path.read_text()
    flat = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", raw))
    for w in REQUIRED:
        ok = w in raw or w in flat
        print(f"  {'✅' if ok else '❌'} 必須含「{w}」")
        if not ok: bad.append(f"缺「{w}」")
    for w in FORBIDDEN:
        ok = w not in raw and w not in flat
        print(f"  {'✅' if ok else '❌'} 不得含「{w}」")
        if not ok: bad.append(f"含「{w}」")
    n_orange = raw.count(f'fill="{ORANGE_FILL}"')
    print(f"  {'✅' if n_orange == 1 else '❌'} 橘色盒數 = {n_orange}（須為 1）")
    if n_orange != 1: bad.append(f"橘色盒 {n_orange} 個")
    n_tiles = raw.count('class="evidence-tile"')
    print(f"  {'✅' if n_tiles == 8 else '❌'} 證據方塊 = {n_tiles}（須為 8）")
    if n_tiles != 8: bad.append(f"證據方塊 {n_tiles}")
    # 三條虛線箭頭終點是否落在 selector 盒內
    sx, sw_ = BOX["sel"]
    ends = re.findall(r'd="M[\d.]+,[\d.]+ L([\d.]+),([\d.]+)"[^>]*stroke-dasharray[^>]*class="to-selector"',
                      raw)
    inside = [sx <= float(x) <= sx + sw_ and SEL_Y <= float(y) <= SEL_Y + SEL_H + 1
              for x, y in ends]
    ok = len(ends) == 3 and all(inside)
    print(f"  {'✅' if ok else '❌'} 虛線箭頭終點在 selector：{sum(inside)}/{len(ends)}（須 3/3）")
    if not ok: bad.append("虛線箭頭終點")
    sizes = [float(s) for s in re.findall(r'font-size="([\d.]+)"', raw)]
    print(f"  {'✅' if min(sizes) >= 8.5 else '❌'} 最小字級 {min(sizes)} pt（須 ≥ 8.5）")
    if min(sizes) < 8.5: bad.append(f"字級 {min(sizes)}")
    if pdf_path.exists():
        out = subprocess.run(["pdffonts", str(pdf_path)], capture_output=True, text=True).stdout
        lines_ = [l for l in out.splitlines()[2:] if l.strip()]
        rows = [l.split() for l in lines_]
        # 欄位由右數：... emb sub uni objID gen → emb = r[-5]（type 欄可能含空格，不可由左數）
        noemb = [r[0] for r in rows if len(r) >= 7 and r[-5] == "no"]
        t3 = [l.split()[0] for l in lines_ if "Type 3" in l]
        print(f"  {'✅' if not noemb else '❌'} PDF 字型全內嵌（未嵌 {len(noemb)}）")
        print(f"  {'✅' if not t3 else '❌'} PDF 無 Type 3（{len(t3)}）；字型：{[r[0].split('+')[-1] for r in rows]}")
        if noemb: bad.append("字型未內嵌")
        if t3: bad.append("Type 3")
    return bad


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="只守門，不重生")
    ap.add_argument("--png-width", type=int, default=2400,
                    help="PNG 像素寬（7.0 in → 2400 px ≈ 343 dpi）")
    ap.add_argument("--work", default=None,
                    help="Chrome 用的暫存 HTML 放哪（預設與輸出同目錄，用完即刪）")
    a = ap.parse_args(argv)
    svg, pdf, png = (OUT_DIR / f"{STEM}.{e}" for e in ("svg", "pdf", "png"))
    if not a.check:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        svg.write_text(build())
        work = Path(a.work) if a.work else OUT_DIR
        svg_to_pdf_chrome(svg, pdf, work)
        subprocess.run(["rsvg-convert", "-f", "png", "-w", str(a.png_width), "-o", str(png),
                        str(svg)], check=True)
        (work / "fig1.html").unlink(missing_ok=True)
        for p in (svg, pdf, png):
            print(f"→ {p}（{p.stat().st_size:,} B）")
    print("── 守門 ──")
    bad = gate(svg, pdf)
    print("✅ 全部通過" if not bad else f"❌ 未通過：{bad}")
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
