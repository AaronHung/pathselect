"""Decision Ledger 的完整性把關（docs/ledger/）。

規則 5：INDEX 是唯一入口 —— 表裡的每個 DR 編號都必須有對應檔案，反之亦然。
規則 3：狀態只有三種 —— ACTIVE / SUPERSEDED-BY DR-0xx / PARKED。
規則 2：一個決策一張卡，固定欄位。
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
LEDGER = REPO_ROOT / "docs" / "ledger"
INDEX = LEDGER / "INDEX.md"
GRAVEYARD = LEDGER / "GRAVEYARD.md"

#: 規則 3 —— 只有這三種狀態
STATUS_RE = re.compile(r"^(ACTIVE|PARKED|SUPERSEDED-BY DR-\d{3})$")
#: INDEX 表的資料列：| [001](DR-001.md) | 標題 | status | 一句話 |
INDEX_ROW_RE = re.compile(
    r"^\|\s*\[(?P<num>\d{3})\]\(DR-(?P=num)\.md\)\s*\|(?P<title>[^|]*)\|"
    r"(?P<status>[^|]*)\|(?P<oneline>.*)\|\s*$")
#: 規則 2 的固定欄位
CARD_FIELDS = ("date:", "status:", "context:", "options:", "ruling:", "evidence:")


def index_rows() -> list[dict]:
    rows = []
    for line in INDEX.read_text(encoding="utf-8").splitlines():
        m = INDEX_ROW_RE.match(line.strip())
        if m:
            rows.append({"num": m.group("num"),
                         "title": m.group("title").strip(),
                         "status": m.group("status").strip(),
                         "oneline": m.group("oneline").strip()})
    return rows


def card_paths() -> list[Path]:
    return sorted(LEDGER.glob("DR-*.md"))


def test_ledger_layout_exists():
    """三層結構：INDEX（L1）、DR 卡（L2）、GRAVEYARD（L3）。"""
    assert LEDGER.is_dir(), "docs/ledger/ 不存在"
    assert INDEX.is_file() and GRAVEYARD.is_file()
    assert card_paths(), "一張 DR 卡都沒有"


def test_index_is_parseable_and_non_empty():
    rows = index_rows()
    assert rows, "INDEX.md 解析不到任何 DR 列（表格格式壞了？）"
    nums = [r["num"] for r in rows]
    assert len(nums) == len(set(nums)), f"INDEX 有重複的 DR 編號：{nums}"


def test_every_index_entry_has_a_card():
    """規則 5：INDEX 的每個 DR 編號必須有對應檔案。"""
    missing = [r["num"] for r in index_rows()
               if not (LEDGER / f"DR-{r['num']}.md").is_file()]
    assert not missing, f"INDEX 列到但沒有卡片：{missing}"


def test_dr_numbers_have_no_gaps():
    """DR 編號不得有缺口。

    append-only 的範圍是「已寫的卡不改內文」，**不是**「不得補記早先的決策」。
    補記是允許且被鼓勵的；缺口代表有裁定沒建檔。
    """
    nums = sorted(int(r["num"]) for r in index_rows())
    assert nums, "INDEX 沒有任何 DR"
    expected = list(range(1, max(nums) + 1))
    missing = sorted(set(expected) - set(nums))
    assert not missing, f"DR 編號有缺口：{[f'DR-{n:03d}' for n in missing]}"
    assert nums[0] == 1, f"編號應從 DR-001 開始，實得 DR-{nums[0]:03d}"


def test_seeds_file_exists_and_is_parseable():
    """SEEDS 是 L3 的另一半：墓園收「決定不做」，種子收「還不知道」。"""
    seeds = LEDGER / "SEEDS.md"
    assert seeds.is_file(), "docs/ledger/SEEDS.md 不存在"
    text = seeds.read_text(encoding="utf-8")
    ids = re.findall(r"^(?:\| )?(S-\d{2})\b", text, re.MULTILINE)
    ids += re.findall(r"^##\s+(S-\d{2})\b", text, re.MULTILINE)
    ids = sorted(set(ids))
    assert ids, "SEEDS.md 解析不到任何 S 編號"
    nums = sorted(int(i.split("-")[1]) for i in ids)
    missing = sorted(set(range(1, max(nums) + 1)) - set(nums))
    assert not missing, f"種子編號有缺口：{[f'S-{n:02d}' for n in missing]}"


def test_every_card_is_listed_in_the_index():
    """反向：不能有卡片沒被 INDEX 收錄（否則 L1 不再是完整入口）。"""
    listed = {r["num"] for r in index_rows()}
    on_disk = {p.stem.split("-")[1] for p in card_paths()}
    assert on_disk == listed, (f"只在磁碟 {sorted(on_disk - listed)}、"
                               f"只在 INDEX {sorted(listed - on_disk)}")


@pytest.mark.parametrize("row", index_rows(), ids=lambda r: f"DR-{r['num']}")
def test_index_status_is_one_of_the_three(row):
    """規則 3：ACTIVE / SUPERSEDED-BY DR-0xx / PARKED，沒有第四種。"""
    assert STATUS_RE.match(row["status"]), (
        f"DR-{row['num']} 的 status {row['status']!r} 不合法；"
        f"只允許 ACTIVE / PARKED / 'SUPERSEDED-BY DR-0xx'")


@pytest.mark.parametrize("path", card_paths(), ids=lambda p: p.stem)
def test_card_has_the_required_fields(path):
    """規則 2：固定欄位一個都不能少。"""
    text = path.read_text(encoding="utf-8")
    missing = [f for f in CARD_FIELDS if f not in text]
    assert not missing, f"{path.name} 缺欄位 {missing}"
    assert text.startswith(f"# {path.stem}  "), f"{path.name} 的標題行格式不符"


@pytest.mark.parametrize("path", card_paths(), ids=lambda p: p.stem)
def test_card_status_matches_the_index(path):
    """卡片 header 的 status 必須與 INDEX 一致，且同樣只有三種。"""
    num = path.stem.split("-")[1]
    row = next((r for r in index_rows() if r["num"] == num), None)
    assert row is not None, f"{path.name} 不在 INDEX 裡"
    # 規則 2 的 header 是「date: ...    status: ...」同一行，status 不在行首
    m = re.search(r"status:\s*(.+?)\s*$", path.read_text(encoding="utf-8"),
                  re.MULTILINE)
    assert m, f"{path.name} 沒有 status 行"
    status = m.group(1).strip()
    assert STATUS_RE.match(status), f"{path.name} 的 status {status!r} 不合法"
    assert status == row["status"], (
        f"{path.name} 的 status {status!r} 與 INDEX 的 {row['status']!r} 不一致")


def test_superseded_cards_point_at_an_existing_card():
    """SUPERSEDED-BY 必須指向真的存在的卡（雙向連結的前提）。"""
    for row in index_rows():
        m = re.match(r"^SUPERSEDED-BY DR-(\d{3})$", row["status"])
        if m:
            assert (LEDGER / f"DR-{m.group(1)}.md").is_file(), (
                f"DR-{row['num']} 指向不存在的 DR-{m.group(1)}")


def test_graveyard_rows_have_a_revival_condition():
    """規則 3：PARKED / 墓園的東西必須寫復活條件，否則就是偷偷刪除。"""
    rows = [l for l in GRAVEYARD.read_text(encoding="utf-8").splitlines()
            if re.match(r"^\|\s*G-\d{2}\s*\|", l)]
    assert rows, "GRAVEYARD.md 解析不到任何 G 列"
    for line in rows:
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        assert len(cells) == 4, f"墓園列欄數不對：{line}"
        assert cells[3], f"{cells[0]} 沒有寫復活條件"
