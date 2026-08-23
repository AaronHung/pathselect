# 測試開發準則

## 1. 新增的斷言必須先被證偽過（mutation check）

**每一條新增的 assert，在提交之前必須以人工製造的違例驗證它真的會失敗，再還原。**

```
1. 寫好斷言
2. 手動破壞它應該保護的東西（刪一列、改一個值、拿掉一個欄位…）
3. 跑測試，確認它 FAIL 且錯誤訊息指得出問題在哪
4. 還原
5. 再跑一次，確認 PASS
```

**永遠為真的斷言是裝飾品，而且比沒有測試更糟 —— 它會給人虛假的安全感。**

實例（`tests/test_ledger.py::test_dr_numbers_have_no_gaps`）：寫完之後把
`INDEX.md` 的 DR-018 那一列移除，確認測試失敗並在訊息中指出
`DR 編號有缺口：['DR-018']`，然後還原。若當時沒做這步，就不會知道
INDEX 的表格解析正則是否真的抓得到那一列。

### 提交訊息要求

提交訊息中以**一行**註明已做過 mutation check 與製造的違例為何：

```
mutation check: 移除 INDEX 的 DR-018 那列 → test_dr_numbers_have_no_gaps FAIL，已還原
```

該次提交沒有新增斷言時，寫 `mutation check: 本次未新增斷言`。

### 範圍

- **只對 2026-08-24 之後新增的斷言要求。** 既有的測試不回頭補。
- **不建自動化 mutation testing 框架。** 時程不允許；人工驗證 + 提交註記就夠。
  這是一條紀律，不是一個工具。

## 2. 執行

```bash
python -m pytest tests/ -q
```

測試不需要 GPU。需要資料集或 CONCH checkpoint 的測試會在缺少時 `skip`，
不會失敗。
