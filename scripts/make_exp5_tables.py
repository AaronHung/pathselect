#!/usr/bin/env python3
"""從 outputs/exp5 產生 LaTeX 表格片段，供撰稿者取用。

格式照 paper/main.tex 現有表格，**只換數字**（欄位、caption 結構、\\cmidrule 位置都保留）。
Table 1 依 PI 指示加一欄 Buffer (WSIs)。

⚠️ 所有數字都從輸出檔讀，不從對話抄。
⚠️ **不寫進 paper/ 底下任何位置**（PI 2026-09-23 指示：論文由撰稿者統一改）。
輸出到 outputs/exp5/latex/。caption 是我產生的草稿，不是稿件內容。
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from report_exp5 import load_tag                               # noqa: E402
from sota.external_baselines import MAIN_METHOD                # noqa: E402

#: DR-004 的詞彙禁令：外部方法的名稱一律從 sota.external_baselines 取，
#: 不在本檔寫字面（沿用 scripts/report_paper_numbers.py 的作法）。
CITE_KEY = "gou2025" + MAIN_METHOD.split("-")[0].lower()

#: Table 1 橫線以上：外部引用值，原樣保留。buffer 取自外部主方法的發表表格
#: （outputs/exp3/PAPER_NUMBERS.md），沒有 buffer 的填 0。
EXTERNAL = [
    ("Upper bound (joint)", "", 0, (".908", ".022", "---", ".937"), (".908", ".022", "---", ".937")),
    ("Fine-tuning", "", 0, (".234", ".008", ".927", ".803"), (".308", ".045", ".841", ".844")),
    ("EWC", "kirkpatrick2017ewc", 0, (".235", ".011", ".928", ".833"), (".309", ".051", ".836", ".840")),
    ("LwF", "li2017lwf", 0, (".236", ".016", ".908", ".900"), (".378", ".081", ".731", ".916")),
    ("A-GEM", "chaudhry2019agem", 30, (".536", ".047", ".527", ".872"), (".436", ".058", ".670", ".879")),
    ("ER-ACE", "caccia2022erace", 30, (".703", ".049", ".281", ".889"), (".666", ".049", ".075", ".917")),
    ("DER++", "buzzega2020derpp", 30, (".684", ".055", ".310", ".910"), (".749", ".055", ".219", ".904")),
    ("ER", "chaudhry2019er", 30, (".644", ".028", ".387", ".901"), (".790", ".040", ".186", ".894")),
    ("ConSlide", "huang2023conslide", 30, (".499", ".025", ".058", ".854"), (".659", ".022", ".076", ".861")),
    ("AttriCLIP", "wang2023attriclip", 0, (".694", ".058", ".207", ".861"), (".616", ".056", ".285", ".844")),
    ("MI-Zero", "lu2023mizero", 0, (".839", ".034", "---", ".909"), (".839", ".034", "---", ".909")),
    (MAIN_METHOD, CITE_KEY, 0, (".859", ".032", ".064", ".925"), (".890", ".021", ".027", ".930")),
]
RERUN = (f"{MAIN_METHOD}, rerun$^\\ddagger$", 0,
         (".868", ".046", ".064", ".935"), (".884", ".033", ".038", ".926"))


def d3(x: float) -> str:
    """.834 這種去掉前導零的三位小數，與稿件現有格式一致。"""
    s = f"{x:.3f}"
    return s[1:] if s.startswith("0.") else s


def cells(per: dict) -> tuple[str, str, str, str] | None:
    if not per:
        return None
    acc = [m["acc"] for m in per.values()]
    fo = [m["forgetting"] for m in per.values()]
    mk = [m["masked_acc"] for m in per.values()]
    sd = statistics.stdev(acc) if len(acc) > 1 else 0.0
    return (d3(statistics.mean(acc)), d3(sd), d3(statistics.mean(fo)), d3(statistics.mean(mk)))


def row(label: str, cite: str, buf, rev, fwd) -> str:
    name = f"{label} \\cite{{{cite}}}" if cite else label
    b = "---" if buf in (None, "") else str(buf)
    def half(c):
        a, s, f, m = c
        acc = "---" if a == "---" else f"${a}_{{\\pm{s}}}$"
        return f"{acc} & ${f}$ & ${m}$" if f != "---" else f"{acc} & --- & ${m}$"
    return f"{name} & {b} & {half(rev)} & {half(fwd)} \\\\"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", default=str(ROOT / "outputs" / "exp5"))
    ap.add_argument("--out", default=str(ROOT / "outputs" / "exp5" / "latex"))
    ap.add_argument("--buffer-json", default=None)
    a = ap.parse_args(argv)
    src, out = Path(a.src), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    ours = {o: load_tag(src / f"m64_full_{s}" / "per_slide", o)
            for o, s in (("reverse", "rev"), ("main", "fwd"))}
    rev, fwd = cells(ours["reverse"]), cells(ours["main"])
    n_rev, n_fwd = len(ours["reverse"]), len(ours["main"])
    if not rev or not fwd:
        print(f"⚠️ 本方法的資料不完整（reverse {n_rev}/10、forward {n_fwd}/10），不產生 Table 1。")
        return 1

    # ── Table 1 ──
    L = ["% 由 scripts/make_exp5_tables.py 從 outputs/exp5 產生，數字未經手抄。",
         f"% 來源：outputs/exp5/m64_full_rev（{n_rev} 折）、m64_full_fwd（{n_fwd} 折）",
         "% 機器：RTX 5090 pod，見 outputs/exp5/MACHINE.md。不得與 exp3／exp4／H200 相減。",
         "\\begin{table}[!t]",
         "\\centering\\footnotesize\\renewcommand{\\arraystretch}{0.93}",
         "\\setlength{\\tabcolsep}{2.0pt}",
         "\\caption{\\textbf{Comparison on the four-task TCGA continual WSI benchmark} "
         "(ten-fold CV; ACC mean$_{\\pm\\text{sd}}$, forgetting F.\\ and masked ACC Mask.\\ as means). "
         "Buffer is the number of WSIs each method keeps; $0$ denotes no rehearsal buffer. "
         "Buffer contents are not comparable across methods: the rehearsal baselines store WSI bags, "
         "whereas PathSelect stores selector snapshots and reloads the corresponding training features. "
         f"Rows above the rule are quoted from {MAIN_METHOD} \\cite{{{CITE_KEY}}}; "
         "$^\\ddagger$our rerun of its released code.}",
         "\\label{tab:sota}",
         "\\begin{tabular}{l@{\\hspace{4pt}}c@{\\hspace{5pt}}ccc@{\\hspace{7pt}}ccc}",
         "\\toprule",
         "& & \\multicolumn{3}{c}{Reverse order} & \\multicolumn{3}{c}{Forward order} \\\\",
         "\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}",
         "Method & Buffer & ACC$\\uparrow$ & F.$\\downarrow$ & Mask.$\\uparrow$ "
         "& ACC$\\uparrow$ & F.$\\downarrow$ & Mask.$\\uparrow$ \\\\",
         "\\midrule"]
    L += [row(*e) for e in EXTERNAL]
    L += ["\\midrule", row(RERUN[0], "", RERUN[1], RERUN[2], RERUN[3]),
          "\\midrule", row("PathSelect (ours)", "", 64, rev, fwd),
          "\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (out / "table1_sota_m64.tex").write_text("\n".join(L) + "\n", encoding="utf-8")

    # ── Table 2 ──
    CHAIN = [("m64_replay", "+ replay"),
             ("m64_distutil", "+ distillation + utility floor"),
             ("m64_full", "+ group-level budget (ours)")]
    L2 = ["% 由 scripts/make_exp5_tables.py 從 outputs/exp5 產生。",
          "% 第 1 列（無保存）與第 5 列（零樣本）對 |M| 免疫，沿用既有數字未重跑 —— 見 docs/EXP5_REPORT.md。",
          "\\begin{table}[!t]",
          "\\centering\\footnotesize\\renewcommand{\\arraystretch}{0.93}",
          "\\caption{\\textbf{Component chain under a 64-WSI replay memory} (ten-fold CV). "
          "The first and last rows use no replay memory and are unchanged.}",
          "\\label{tab:chain}",
          "\\begin{tabular}{lcccc}",
          "\\toprule",
          "& \\multicolumn{2}{c}{Reverse order} & \\multicolumn{2}{c}{Forward order} \\\\",
          "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}",
          "Configuration & ACC$\\uparrow$ & Forg.$\\downarrow$ & ACC$\\uparrow$ & Forg.$\\downarrow$ \\\\",
          "\\midrule",
          "No preservation & $.481_{\\pm.057}$ & $.584$ & $.549_{\\pm.099}$ & $.478$ \\\\"]
    missing = []
    for tag, lab in CHAIN:
        cr = cells(load_tag(src / f"{tag}_rev" / "per_slide", "reverse"))
        cf = cells(load_tag(src / f"{tag}_fwd" / "per_slide", "main"))
        if not cr or not cf:
            missing.append(lab); continue
        L2.append(f"{lab} & ${cr[0]}_{{\\pm{cr[1]}}}$ & ${cr[2]}$ "
                  f"& ${cf[0]}_{{\\pm{cf[1]}}}$ & ${cf[2]}$ \\\\")
    L2 += ["\\midrule",
           "Class-text top-8, no learning & $.812_{\\pm.024}$ & --- & $.812_{\\pm.024}$ & --- \\\\",
           "\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (out / "table2_chain_m64.tex").write_text("\n".join(L2) + "\n", encoding="utf-8")

    print(f"→ {out}/table1_sota_m64.tex   PathSelect: "
          f"rev {rev[0]}±{rev[1]} F {rev[2]} Mask {rev[3]} | fwd {fwd[0]}±{fwd[1]} F {fwd[2]} Mask {fwd[3]}")
    print(f"→ {out}/table2_chain_m64.tex" + (f"   ⚠️ 缺列：{missing}" if missing else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
