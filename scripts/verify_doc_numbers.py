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

#: G345 的對照組是 G1' 的 hier-A5
BASELINE = ("exp2/hier2", "A5", "hier")

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


def load(sub: str, arm: str, arch: str) -> list[dict]:
    d = OUT / sub / "per_slide"
    return [r for f in sorted(d.glob("*.json")) for r in json.loads(f.read_text())
            if r["arm"] == arm and r.get("arch") == arch
            and r.get("allocation") == "per_budget" and r.get("order") == "reverse"]


def metrics(recs, arm, label_space):
    seeds = sorted({r["seed"] for r in recs})
    return seeds, {s: arm_metrics(recs, arm, TASKS, s, label_space) for s in seeds}


def recompute(sub, arm, arch, key, label_space):
    """回傳 (mean, std, wins, n)，全部從 per_slide 重算。"""
    a, b = load(sub, arm, arch), load(*BASELINE)
    if not a or not b:
        return None
    sa, Ma = metrics(a, arm, label_space)
    sb, Mb = metrics(b, BASELINE[1], label_space)
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
            hit = any(any(v in norm(ln) for v in want) for ln in lines)
            print(f"  {'✅' if hit else '❌'} {label:12s} 重算 {shown}  ← {doc}")
            if not hit:
                bad.append(f"{label}：{doc} 的「{anchor}」該行沒有重算值 {want}")

    print("── Tier 2：文件寫的數字必須**同時**出現在被引用的產物裡 ──")
    for label, doc, text, art in QUOTED:
        dt = (ROOT / doc).read_text(encoding="utf-8")
        ap = OUT / art
        at = ap.read_text(encoding="utf-8") if ap.exists() else ""
        in_doc, in_art = text in dt, text in at
        ok = in_doc and in_art
        print(f"  {'✅' if ok else '❌'} {label:22s} {text:14s} "
              f"文件={'有' if in_doc else '**無**'} 產物={'有' if in_art else '**無**'}")
        if not ok:
            bad.append(f"{label}：文件={in_doc}、產物={in_art}（兩者都要有）")

    if bad:
        print("\n❌ 對不上：")
        for b in bad:
            print(f"  - {b}")
        return 1
    print("\n✅ 全部可溯源")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
