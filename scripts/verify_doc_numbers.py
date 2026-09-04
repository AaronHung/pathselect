"""治理文件裡的數字必須能從產物重算出來（不憑記憶）。

兩層驗證：

**Tier 1（最強）**：從 `per_slide/*.json` **重新計算**配對統計，再檢查文件裡
寫的數字與重算結果相符。這條路完全不經過任何 .md ——.md 若寫錯，這裡會抓到。

**Tier 2**：文件引用某個產物 .md 的數字時，確認那個字串**確實出現在該產物裡**。
用於年代較久、原始 per_slide 欄位已不足以重算的數字。

任何一個對不上就 exit 1。這是為了讓「總表裡的每個數字都能溯源」變成可執行的檢查，
而不是一次性的人工比對。
"""
from __future__ import annotations

import json
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from run_exp2 import ORDERS, arm_metrics                             # noqa: E402
from selector.text_encoder import load_config                        # noqa: E402

OUT = ROOT / "outputs"
TASKS = ORDERS["reverse"]

#: 每筆 = (說明, per_slide 目錄, 臂, arch, 指標, [(文件, 該行的唯一錨點)])
#:
#: ⚠️ 錨點是**行級**的：重算值必須出現在含該錨點的那一行。
#:    第一版用整份文件的子字串搜尋 —— 「5.92」在別處出現就算通過，
#:    把數字改錯完全抓不到。行級錨定才守得住。
PAIRED = [
    ("G5 task-IL", "exp2/arch", "A5", "hier_state", "final_task_il", [
        ("docs/ledger/DR-043.md", "| G5 state |"),
        ("docs/CLAIMS.md", "但依 pre-registered 判準**落判 FAIL**"),
    ]),
    ("G5 class-IL", "exp2/arch", "A5", "hier_state", "final_class_il", [
        ("docs/ledger/DR-043.md", "| G5 state |"),
        ("docs/CLAIMS.md", "但依 pre-registered 判準**落判 FAIL**"),
    ]),
    ("G4 task-IL", "exp2/arch", "A5", "hier_query", "final_task_il", [
        ("docs/ledger/DR-043.md", "| G4 q_tau |"),
        ("docs/CLAIMS.md", "依 pre-registered 雙軸判準**落判 FAIL**"),
    ]),
    ("G4 class-IL", "exp2/arch", "A5", "hier_query", "final_class_il", [
        ("docs/ledger/DR-043.md", "| G4 q_tau |"),
        ("docs/CLAIMS.md", "依 pre-registered 雙軸判準**落判 FAIL**"),
    ]),
    ("G4 洩漏率", "exp2/arch", "A5", "hier_query", "mean_leak", [
        ("docs/ledger/DR-043.md", "| G4 | 跨任務洩漏率 |"),
        ("docs/CLAIMS.md", "q_tau 使跨任務洩漏率系統性降低"),
    ]),
    ("G3 task-IL", "exp2/arch", "A5g", "hier", "final_task_il", [
        ("docs/ledger/DR-043.md", "| G3 group L_sem |"),
        ("docs/CLAIMS.md", "G3 落判 FAIL（DR-043）"),
    ]),
    # C-27 只寫 task-IL 與 class-IL 的 win count；量級寫在 C-29 的條件段裡。
    ("G3 class-IL", "exp2/arch", "A5g", "hier", "final_class_il", [
        ("docs/ledger/DR-043.md", "| G3 group L_sem |"),
        ("docs/CLAIMS.md", "**必須同時附上條件：未轉化為準確率增益。** task-IL −0.524"),
    ]),
    ("G3 配額 KL", "exp2/arch", "A5g", "hier", "mean_quota_kl", [
        ("docs/ledger/DR-043.md", "| G3 | group 配額 KL |"),
        ("docs/CLAIMS.md", "group 層語意先驗使 group 配額分佈的 KL"),
    ]),
]

#: 預設對照組：G1' 的 hier-A5。PAIRED 的第 6 欄可覆寫成任意 (dir, arm, arch)。
BASELINE = ("exp2/hier2", "A5", "hier")

#: 額外的 Tier 1 配對（PROMPT DOSSIER-FIGURES §A4）：對照組不是預設 baseline 的組合。
#: (說明, dirA, armA, archA, key, [(文件, 錨點)], (dirB, armB, archB))
PAIRED_CUSTOM = [
    ("hier A5−A3 task-IL", "exp2/hier2", "A5", "hier", "final_task_il",
     [("docs/RESULTS_DOSSIER.md", "| **hier-A5 − hier-A3** |")],
     ("exp2/hier2", "A3", "hier")),
    ("hier A5−A3 class-IL", "exp2/hier2", "A5", "hier", "final_class_il",
     [("docs/RESULTS_DOSSIER.md", "| **hier-A5 − hier-A3** |")],
     ("exp2/hier2", "A3", "hier")),
    ("group-KD task-IL", "exp2/hier2", "A5", "hier", "final_task_il",
     [("docs/RESULTS_DOSSIER.md", "| task-IL | **+3.95 ± 2.50** |")],
     ("exp2/hier2", "A5nG", "hier")),
    ("group-KD class-IL", "exp2/hier2", "A5", "hier", "final_class_il",
     [("docs/RESULTS_DOSSIER.md", "| class-IL | **+3.71 ± 2.66** |")],
     ("exp2/hier2", "A5nG", "hier")),
]

#: Tier 2：(說明, 文件, 該數字, 被引用的產物)
#: ⚠️ 條件是 **文件有 且 產物也有**。第一版寫成 (not in_doc) or in_art ——
#:    文件把數字改掉後 not in_doc 為真，檢查會空洞地通過。
QUOTED = [
    ("group-KD task-IL", "docs/CLAIMS.md", "+3.95 ± 2.50", "exp2/hier2/HIER2.md"),
    ("group-KD class-IL", "docs/CLAIMS.md", "+3.71 ± 2.66", "exp2/hier2/HIER2.md"),
    ("階層 A5−A3 task-IL", "docs/PROJECT_NARRATIVE.md", "+3.28 ± 2.40",
     "exp2/hier2/HIER2.md"),
    ("flat A5−A3 task-IL", "docs/PROJECT_NARRATIVE.md", "+0.74 ± 1.93",
     "exp2/main/EXP2.md"),
    ("A3@256−A3@512", "docs/PROJECT_NARRATIVE.md", "+4.25", "exp2/memory/MEMORY.md"),
    ("跨容量 A5@128−A3@512", "docs/PROJECT_NARRATIVE.md", "+3.06 ± 2.80",
     "exp2/memory_hier/MEMORY_HIER.md"),
]

#: RESULTS_DOSSIER 的 38 條（PROMPT DOSSIER-FIGURES-20260826 §A2）。
#: 第 5 欄（可選）= 文件端的字串，只在總表刻意寫縮寫時登記。**不是容忍，是明列。**
#: ⚠️ 總表用 U+2212（−），產物用 ASCII（-）。**正規化在檢查器裡做，不改總表的數字。**
DOSSIER = "docs/RESULTS_DOSSIER.md"
QUOTED += [
    ("Exp0 K=1 差距", DOSSIER, "+34.48", "exp0/BASELINES.md"),
    ("Exp0 K=8 差距", DOSSIER, "+21.90", "exp0/BASELINES.md"),
    ("Exp0 K=64 差距", DOSSIER, "+20.14", "exp0/BASELINES.md"),
    ("Exp0 峰值", DOSSIER, "0.8797", "exp0/BASELINES.md"),
    ("eff_K/K @64", DOSSIER, "0.375", "exp0/EFFECTIVE_K.md"),
    ("S1 slide 平均", DOSSIER, "0.9821", "exp1/diag/TASK_SEPARABILITY.md"),
    ("S1 prototype", DOSSIER, "0.9857", "exp1/diag/TASK_SEPARABILITY.md"),
    ("S1 patch", DOSSIER, "0.8930", "exp1/diag/TASK_SEPARABILITY.md"),
    ("A1 class-IL forgetting", DOSSIER, "+51.93", "exp2/main/EXP2.md"),
    ("A1 task-IL forgetting", DOSSIER, "+11.09", "exp2/main/EXP2.md"),
    ("A1 洩漏率", DOSSIER, "0.4408", "exp2/main/EXP2.md"),
    ("A1 esca 洩漏 @T4", DOSSIER, "0.7467", "exp2/main/EXP2.md"),
    ("A5−A3 flat task-IL", DOSSIER, "+0.74 ± 1.93", "exp2/main/EXP2.md"),
    ("A5−A3 flat class-IL", DOSSIER, "+4.61 ± 2.29", "exp2/main/EXP2.md"),
    ("A4−A3 task-IL", DOSSIER, "-1.04 ± 0.15", "exp2/main/EXP2.md"),
    ("A5−B1 class-IL", DOSSIER, "+22.40 ± 13.06", "exp2/ablation/EXP2.md", "+22.40"),
    ("B1 洩漏率", DOSSIER, "0.3231", "exp2/ablation/B1_LANDING.md"),
    ("β_u 配對", DOSSIER, "+1.12 ± 0.50", "exp2/ablation/BETA_U.md"),
    ("G1 判準", DOSSIER, "-18.69 ± 10.41", "exp2/hier/HIER.md"),
    ("hier A5−A3 task-IL", DOSSIER, "+3.28 ± 2.40", "exp2/hier2/HIER2.md"),
    ("hier A5−A3 class-IL", DOSSIER, "+5.76 ± 3.42", "exp2/hier2/HIER2.md"),
    ("hier A5 洩漏率差", DOSSIER, "+2.15 ± 1.96", "exp2/hier2/HIER2.md"),
    ("hier-A3−flat-A3", DOSSIER, "-3.11 ± 3.42", "exp2/hier2/HIER2.md"),
    ("group-KD task-IL（總表）", DOSSIER, "+3.95 ± 2.50", "exp2/hier2/HIER2.md"),
    ("group-KD class-IL（總表）", DOSSIER, "+3.71 ± 2.66", "exp2/hier2/HIER2.md"),
    ("L_sem disc−none", DOSSIER, "-0.02 ± 0.89", "exp2/prior/PRIOR.md"),
    ("L_sem disc−max_sim", DOSSIER, "+0.76 ± 0.75", "exp2/prior/PRIOR.md"),
    ("記憶體 64", DOSSIER, "+4.90 ± 2.67", "exp2/memory_hier/MEMORY_HIER.md"),
    ("記憶體 1024", DOSSIER, "+2.49 ± 2.42", "exp2/memory_hier/MEMORY_HIER.md"),
    ("4× 配對", DOSSIER, "+3.06 ± 2.80", "exp2/memory_hier/MEMORY_HIER.md"),
    ("class-IL 1024", DOSSIER, "+1.88 ± 5.69", "exp2/memory_hier/MEMORY_HIER.md"),
    ("flat 2× 錨點", DOSSIER, "0.8203", "exp2/memory/MEMORY.md"),
    ("flat A3 下滑", DOSSIER, "+4.25 ± 2.27", "exp2/memory/MEMORY.md"),
    ("G5 task-IL", DOSSIER, "-0.497 ± 2.206", "exp2/arch/ARCH_COMPLETENESS.md",
     "-0.50 ± 2.21"),
    ("G4 class-IL", DOSSIER, "+5.849 ± 3.520", "exp2/arch/ARCH_COMPLETENESS.md",
     "+5.85 ± 3.52"),
    ("G4 洩漏率", DOSSIER, "-5.923 ± 4.189", "exp2/arch/ARCH_COMPLETENESS.md",
     "-5.92 ± 4.19"),
    ("G3 配額 KL", DOSSIER, "-0.005 ± 0.004", "exp2/arch/ARCH_COMPLETENESS.md"),
    ("G5 重測重疊", DOSSIER, "2.81/8", "exp2/arch/ARCH_COMPLETENESS.md"),
]


def load(sub: str, arm: str, arch: str) -> list[dict]:
    d = OUT / sub / "per_slide"
    return [r for f in sorted(d.glob("*.json")) for r in json.loads(f.read_text())
            if r["arm"] == arm and r.get("arch") == arch
            and r.get("allocation") == "per_budget" and r.get("order") == "reverse"]


def metrics(recs, arm, label_space):
    seeds = sorted({r["seed"] for r in recs})
    return seeds, {s: arm_metrics(recs, arm, TASKS, s, label_space) for s in seeds}


def recompute(sub, arm, arch, key, label_space, baseline=None):
    """回傳 (mean, std, wins, n)，全部從 per_slide 重算。"""
    a, b = load(sub, arm, arch), load(*(baseline or BASELINE))
    if not a or not b:
        return None
    sa, Ma = metrics(a, arm, label_space)
    sb, Mb = metrics(b, (baseline or BASELINE)[1], label_space)
    common = sorted(set(sa) & set(sb))
    d = [Ma[s][key] - Mb[s][key] for s in common
         if Ma[s].get(key) is not None and Mb[s].get(key) is not None]
    if not d:
        return None
    higher = key not in ("mean_leak", "mean_quota_kl")
    w = sum((x > 0) if higher else (x < 0) for x in d)
    sd = statistics.stdev(d) if len(d) > 1 else 0.0
    return statistics.mean(d), sd, w, len(d)


def norm(t: str) -> str:
    """統一減號：文件用 U+2212，格式化字串用 ASCII。"""
    return t.replace("\u2212", "-").replace("\u2013", "-")


def doc_forms(want: str) -> set[str]:
    """已停用的四捨五入容忍。保留函式以說明為何不用。

    ⚠️ 曾經允許「少一位小數」的寫法，被 mutation 證明危險：
    `-0.005` 的 2 位形式是 `-0.01`，而文件裡另一個不相干的數字（§4.4 的
    Jaccard −0.01）正好長這樣 —— 檢查通過，但通過的理由是錯的。
    模糊比對在治理文件上不可用。**文件端一律要求完全相符**；
    總表刻意寫縮寫時，在 QUOTED 的第 5 欄明確登記文件端字串。
    """
    return {want}


def contains_number(text: str, form: str) -> bool:
    """form 必須以**完整數字**出現，不能只是某個更長數字的前綴。

    ⚠️ 這條是被 mutation 逼出來的：原本用 `form in text`，而 `+21.9` 是
    `+21.95` 的前綴 —— 把總表的數字改一位，較短的四捨五入寫法仍然命中，
    38 條裡有 31 條漏抓。必須要求右邊不接數字、左邊不接數字或小數點。
    """
    return re.search(r"(?<![\d.])" + re.escape(form) + r"(?!\d)", text) is not None


def anchored_lines(path: Path, anchor: str) -> list[str]:
    """回傳含錨點的**整個段落**（錨點行 + 後續非空行）。

    ⚠️ 只取錨點那一行不夠 —— markdown 的一句話常跨行，
    「task-IL +0.312（2/5）、\nclass-IL +5.849…」就是這樣，
    只比對單行會誤報成文件寫錯。
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    out = []
    for i, ln in enumerate(lines):
        if anchor not in ln:
            continue
        j = i
        while j < len(lines) and lines[j].strip():
            out.append(lines[j])
            j += 1
    return out


# ── 論文稿的數值溯源（DR-046 凍結）──────────────────────────────────────────

PAPER = ROOT / "paper" / "main.tex"
#: EXP2.md（flat）是 EXP2_hier.md 的同批對照，配對表在那裡；DR046_GATES 是
#: 本輪新增的 gate 總表。兩者都是 committed artifact，一併納入掃描來源。
PAPER_ARTIFACTS = ["docs/DR046_TABLE.md", "docs/RESULTS_DOSSIER.md",
                   "outputs/exp2/main/EXP2_hier.md", "outputs/exp2/main/EXP2.md",
                   "docs/DR046_GATES.md",
                   # Exp 0 預算掃描（論文要引用 28/28 與 0.8797 @ K=8）
                   "outputs/exp0/BASELINES.md",
                   # DR-048 SOTA 協定線（Prompt 6-5：verify 擴掃納入 sota/ 產物）
                   "docs/SOTA_TABLE.md", "docs/MEMORY_FOOTPRINT.md",
                   "outputs/exp2/sota/EXP2.md"]
PAPER_TOL = 5e-3

#: 稿內合法但**不是實驗結果**的數字，逐個列出理由。不得用來塞不會溯源的結果值。
PAPER_ALLOW = {
    "2.1": "rank-4 LoRA 佔參數比（16,400 / 選擇器總參數），由架構推導，非實驗結果",
}


#: LaTeX 的長度單位與相對寬度。`p{4.75cm}`、`0.48\\textwidth` 之類是**排版參數**，
#: 不是實驗數字，掃描要略過 —— 否則會逼人把版面寬度也塞進「可溯源產物」。
LATEX_LEN = re.compile(
    r"(?:cm|mm|in|pt|bp|pc|dd|cc|sp|ex|em|\\(?:text|line|column|page)width|\\(?:text|page)height)")


def paper_numbers(text: str):
    """稿內的帶小數數字（略過註解行與 LaTeX 長度）。

    整數不看 —— rank 4、5 seeds 之類不是結果值。
    """
    out = []
    for i, ln in enumerate(text.splitlines(), 1):
        if ln.lstrip().startswith("%"):
            continue
        for m in re.finditer(r"[-+\u2212]?\d+\.\d+", ln):
            if LATEX_LEN.match(ln[m.end():].lstrip()[:14]):
                continue                      # 4.75cm / 0.48\textwidth → 排版參數
            out.append((i, m.group(0), ln.strip()))
    return out


def artifact_numbers() -> list[float]:
    pool = []
    for rel in PAPER_ARTIFACTS:
        f = ROOT / rel
        if not f.exists():
            continue
        for m in re.finditer(r"[-+\u2212]?\d+\.\d+", norm(f.read_text(encoding="utf-8"))):
            try:
                pool.append(float(m.group(0)))
            except ValueError:
                pass
    return pool


def traceable(tok: str, pool: list[float]) -> bool:
    """稿內數字能否溯源。

    ⚠️ 容差是「**先對齊稿內的位數**再比 5e-3」，不是對原值取 5e-3 ——
    稿裡寫 `34.0`（1 位）而產物是 `-34.03`，直接比會差 0.03 而誤判。
    同時容許 ×100 / ÷100（產物用比例 0.9821、稿裡用百分比 98.21）。
    """
    v = abs(float(norm(tok)))
    k = len(tok.split(".")[1])
    for y in pool:
        for cand in (abs(y), abs(y) * 100.0, abs(y) / 100.0):
            if abs(round(cand, k) - round(v, k)) <= PAPER_TOL:
                return True
    return False


def scan_paper() -> list[str]:
    """回傳問題清單；空 = 通過。

    ⚠️ **已知弱點：數值池比對會誤配。** 本掃描只問「這個數字在產物裡存不存在」，
    不問「它是不是**這個主張**的那個數字」。實例：稿內的 `+13.2` 命中的是
    `+25.26 ± 13.24` 裡的一個**標準差**，與該句主張毫無關係。
    因此它能抓「憑空捏造的數字」，但抓不到「引用了別處的數字」。
    要更強的保證，需要逐句標註來源，超出本檔範圍。
    """
    if not PAPER.exists():
        return [f"找不到 {PAPER}"]
    text = PAPER.read_text(encoding="utf-8")
    bad = []

    uses = [(i, l) for i, l in enumerate(text.splitlines(), 1)
            if re.search(r"\\pending\{", l) and not l.lstrip().startswith("%")
            and "newcommand" not in l]
    print(f"  \\pending 用例：{len(uses)} " + ("✅" if not uses else "❌"))
    for i, l in uses:
        bad.append(f"paper/main.tex L{i} 還有 \\pending 用例：{l.strip()[:70]}")

    pool = artifact_numbers()
    nums = paper_numbers(text)
    seen, miss = set(), []
    for line_no, tok, ctx in nums:
        key = norm(tok).lstrip("+")
        if key in seen:
            continue
        seen.add(key)
        if key.lstrip("-") in PAPER_ALLOW or key in PAPER_ALLOW:
            continue
        if not traceable(tok, pool):
            miss.append((line_no, tok, ctx))
    print(f"  稿內相異數值 {len(seen)}｜白名單 {len(PAPER_ALLOW)}｜"
          f"無法溯源 {len(miss)} " + ("✅" if not miss else "❌"))
    for line_no, tok, ctx in miss:
        bad.append(f"paper/main.tex L{line_no} 的 {tok} 溯源不到：{ctx[:70]}")
    return bad


def main() -> int:
    cfg = load_config()
    ls = list(cfg["tasks"])
    bad = []

    print("── Tier 1：從 per_slide 重算，再比對文件的**指定行** ──")
    for label, sub, arm, arch, key, locs in PAIRED:
        r = recompute(sub, arm, arch, key, ls)
        if r is None:
            bad.append(f"{label}：找不到 per_slide 資料"); continue
        mean, sd, w, n = r
        sc = 1 if key == "mean_quota_kl" else 100
        want = {f"{mean * sc:+.2f}", f"{mean * sc:+.3f}"}
        shown = f"{mean * sc:+.3f} ± {sd * sc:.3f}（{w}/{n}）"
        for doc, anchor in locs:
            lines = anchored_lines(ROOT / doc, anchor)
            if not lines:
                bad.append(f"{label}：{doc} 找不到錨點「{anchor}」"); 
                print(f"  ❌ {label:12s} {doc}：錨點不存在「{anchor}」")
                continue
            hit = any(any(contains_number(norm(ln), v) for v in want) for ln in lines)
            print(f"  {'✅' if hit else '❌'} {label:12s} 重算 {shown}  ← {doc}")
            if not hit:
                bad.append(f"{label}：{doc} 的「{anchor}」該行沒有重算值 {want}")

    print("── Tier 1b：非預設對照組的配對 ──")
    for label, sub, arm, arch, key, locs, base in PAIRED_CUSTOM:
        r = recompute(sub, arm, arch, key, ls, baseline=base)
        if r is None:
            bad.append(f"{label}：找不到 per_slide 資料"); continue
        mean, sd, w, n = r
        want = {f"{mean * 100:+.2f}", f"{mean * 100:+.3f}"}
        for doc, anchor in locs:
            lines = anchored_lines(ROOT / doc, anchor)
            hit = bool(lines) and any(any(contains_number(norm(ln), v)
                                          for v in want) for ln in lines)
            print(f"  {'✅' if hit else '❌'} {label:20s} 重算 {mean * 100:+.2f} ± "
                  f"{sd * 100:.2f}（{w}/{n}）  ← {doc}")
            if not hit:
                bad.append(f"{label}：{doc} 的「{anchor}」對不上重算值 {want}")

    print("── Tier 2：登記的是**產物原字串**；文件端接受同值的四捨五入寫法 ──")
    for row in QUOTED:
        label, doc, text, art = row[:4]
        dt = norm((ROOT / doc).read_text(encoding="utf-8"))
        ap = OUT / art
        at = norm(ap.read_text(encoding="utf-8")) if ap.exists() else ""
        want = norm(text)
        want_doc = norm(row[4]) if len(row) > 4 else want
        in_art = contains_number(at, want)
        in_doc = contains_number(dt, want_doc)
        ok = in_doc and in_art
        print(f"  {'✅' if ok else '❌'} {label:22s} {text:14s} "
              f"文件={'有' if in_doc else '**無**'} 產物={'有' if in_art else '**無**'}")
        if not ok:
            bad.append(f"{label}：文件={in_doc}、產物={in_art}（兩者都要有）")

    print("── 論文稿溯源：paper/main.tex ──")
    bad += scan_paper()

    if bad:
        print("\n❌ 對不上：")
        for b in bad:
            print(f"  - {b}")
        return 1
    print("\n✅ 全部可溯源")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
