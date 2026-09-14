# paper/versions/

**常規**：`paper/` 根目錄只留 `main.tex`（當前稿，內含 `\markuptrue/\markupfalse` 開關）
與 `main_marked.tex` 包裝；所有備份住這裡。**PDF 不進 git**（`.gitignore`：`paper/*.pdf`），
只有 `paper/figures/*.pdf` 圖檔例外。

| 位置 | 內容 |
|---|---|
| `versions/`（本層） | 2026-09-13 起、有 markup 巨集之後的備份：v0.9_markup_macros、v0.91_g1（=v0.91a）、v0.91b、… |
| `versions/old/` | 2026-09-13 之前的所有備份（v0.5–v0.89s）、`extended_master.tex`、`spconf_pre2027.sty`、`archive/`（舊版 Fig.1）。只存於 GitHub，不同步到 Overleaf；版本地圖見該目錄 README |

入庫慣例：備份與 `main.tex` 位元相同 → `git mv` 進本層；自檢用 clean 版核頁數、marked 版只核零錯誤。
