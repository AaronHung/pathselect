# 論文中隨 replay memory 大小改變的數字（|M| 512 → 64）

盤點於 commit `081bdcd`，分支 `exp/m64-5090-backup`。**唯讀盤點，未改動任何檔案。**
每一項都註明位置、目前數字、依賴設定、來源目錄。

---

## 0. 先講一個會擴大影響範圍的事實

`scripts/run_exp2.py:247` 取樣記憶體的條件是 **`kd` 或 `eq` 或 `replay`** 任一為真，
不是只有 `replay`：

```python
if len(memory) and (spec["kd"] or spec["eq"] or spec["replay"]):
    for entry in memory.sample(args.replay_k, rng):
```

`scripts/run_exp2.py:379` 填記憶體用同一個條件。因此 **B1（只有 KD）與 B2（只有 hinge）
也依賴 |M|**，即使它們的 `replay=False`。任務指示裡「含 replay 的設定」會低估範圍。

**|M| = 512 到底有沒有綁住？** 訓練張數（`docs/RESULTS_DOSSIER.md:129`）：
esca 120、rcc 616、brca 763、lung 774。`fill_memory` 預設寫入該任務**全部**訓練切片。

| 順序 | 學完 task 1 | task 2 | task 3 | task 4 |
|---|---|---|---|---|
| reverse（esca→rcc→brca→lung） | 120（**512 未綁住**） | 736→512 綁住 | 綁住 | 綁住 |
| forward（lung→brca→rcc→esca） | 774→512 綁住 | 綁住 | 綁住 | 綁住 |

**|M| = 64 時兩個順序都從 task 1 起就綁住**，而且每個 stage 的任務組成都會變。
所以這不是二階效應 —— 每一個依賴記憶體的數字都會動。

`mem_capacity` 有寫進每一筆 per_slide 記錄（`run_exp2.py:318`），可事後自檢。

---

## 1. Table 1（`tab:sota`，main.tex:169-198）

橫線以上 12 列是**外部引用**（QPMIL-VL 的 Tab. 1/2），與本方法的 |M| 無關。

| 行 | 列 | reverse ACC / F. / Mask. | forward ACC / F. / Mask. | 依賴 \|M\|？ |
|---|---|---|---|---|
| 180–191 | Upper bound、Fine-tuning、EWC、LwF、A-GEM、ER-ACE、DER++、ER、ConSlide、AttriCLIP、MI-Zero、QPMIL-VL | 見稿 | 見稿 | 外部引用，否 |
| 193 | QPMIL-VL rerun ‡ | .868±.046 / .064 / .935 | .884±.033 / .038 / .926 | 否（對方的程式與 buffer） |
| 195 | **PathSelect (ours)** | **.834±.031 / .101 / .914** | **.837±.027 / .059 / .916** | **是（6 格全動）** |

PathSelect 那列 = A5、hier、accumulating head、`--uold current`、十折。
來源 `outputs/exp3/sota_ucur`，dossier §11.9。

⚠️ `outputs/exp3/PAPER_NUMBERS.md` 的對應列（0.803/0.835）是**舊的** flat + snapshot 版本，
與稿件不符。不要從那份檔案重新產生論文數字。

---

## 2. Table 2（`tab:chain`，main.tex:203-221）

| 行 | 列 | reverse ACC / Forg. | forward ACC / Forg. | arm | 依賴 \|M\|？ |
|---|---|---|---|---|---|
| 214 | No preservation | .481±.057 / .584 | .549±.099 / .478 | **A1** | **否**（記憶體從不填也不取） |
| 215 | + replay | .819±.042 / .121 | .831±.044 / .078 | **A3** | **是** |
| 216 | + distillation + utility floor | .823±.041 / .108 | .846±.047 / .068 | **A5 flat, `--uold current`** | **是** |
| 217 | + group-level budget (ours) | .834±.031 / .101 | .837±.027 / .059 | **A5 hier, `--uold current`** | **是** |
| 218 | Class-text top-8, no learning | .812±.024 / — | .812±.024 / — | 非 run_exp2 的臂（`sota/zeroshot_topb.py`） | **否**（零參數） |

**第 2、3、4 列會變；第 1、5 列不變。**

---

## 3. Table 3（`tab:ablation`，main.tex:227-245）

協定：**fold 1、5 個 seed、flat（單層）、reverse、accumulating head**。

| 行 | 列 | Rep./Dist./Util. | Class-IL / Task-IL / Leak. / Jacc. / ΔU | arm | 依賴 \|M\|？ |
|---|---|---|---|---|---|
| 237 | No preservation | — — — | 44.5 / 81.4 / 48.0 / .002 / −4.36 | **A2** | **否** |
| 238 | | — ✓ — | 57.3 / 84.6 / 34.8 / .049 / −2.99 | **B1** | **是**（KD 取記憶體） |
| 239 | | — — ✓ | 78.9 / 89.3 / 13.1 / .070 / −0.88 | **B2 `--uold current`** | **是**（hinge 取記憶體） |
| 240 | | ✓ — — | 77.0 / 90.2 / 15.0 / .056 / −0.64 | **A3** | **是** |
| 241 | | ✓ ✓ — | 75.6 / 90.6 / 16.2 / .165 / −0.76 | **A4** | **是** |
| 242 | Full preservation | ✓ ✓ ✓ | 79.5 / 90.9 / 12.7 / .125 / −0.57 | **A5 `--uold current`** | **是** |

**六列中五列會變**，只有 A2 不變。

⚠️ Table 3 **混用兩個批次**：第 1/2/4/5 列來自 `outputs/exp3/ablation_acc`（`--uold snapshot`），
第 3/6 列來自 `outputs/exp3/ablation_ucur`（`--uold current`）。caption 沒有揭露這件事。
M=64 的重跑必須同時重現兩個條件，否則表格會以新的方式內部不一致。

---

## 4. 正文與摘要中的數字

| 位置 | 數字 | 敘述 | 依賴 \|M\|？ |
|---|---|---|---|
| Abstract :62 | 0.481 → **0.834** | 保存機制提升 class-IL，reverse | 起點否、**終點是** |
| Abstract :62 | 0.549 → **0.837** | forward | 起點否、**終點是** |
| Abstract :62 | 「保住證據身分 ≠ 保住有用性」 | 靠 Table 3 的 Jaccard 反轉 | **是** |
| §1 :90 | 0.48 → **0.83**、0.55 → **0.84** | 摘要數字的四捨五入重述 | 同上 |
| §1 :94 | 16.4k | LoRA 參數 | 否 |
| **§1 :96** | **512 snapshots** | 「Its capacity is fixed at 512 snapshots」 | **要改成 64** |
| §4.1 :167 | CONCH **512**-dimensional | 特徵維度 | **否 —— 改數字時不要動到這個** |
| **§4.1 :167** | **at most 512 Selection Memory snapshots** | 協定敘述 | **要改成 64** |
| §4.2 :223 | 0.481 / 0.584、0.549 / 0.478 | SeqFT | 否（A1） |
| §4.2 :223 | Jaccard 0.002、ΔU −4.36 | A2 | 否 |
| §4.2 :223 | 「vs −0.64 or better」 | A3 的 ΔU | **是** |
| §4.2 :225 | 0.481→**0.819**、0.549→**0.831**、「每一折都成立」 | replay 恢復 | **是** |
| §4.2 :225 | **0.823**、**0.846** | A5 flat ucur | **是** |
| §4.2 :225 | **0.834**、**0.837** | A5 hier ucur | **是** |
| §4.2 :225 | **+0.015**、**+0.006**，**6/10**、**7/10** | A5 hier − A3 | **是** |
| §4.2 :225 | **2.5** 個 class-IL 點、**5/5** seeds | fold 1 的 A5_ucur − A3 | **是** |
| §4.2 :247 | 洩漏 **34.8** | B1 | **是** |
| §4.2 :247 | **78.9**、**77.0** | B2_ucur、A3 | **是** |
| §4.2 :247 | **79.5** | A5_ucur | **是** |
| §4.2 :247 | Jaccard **0.165** 對 **0.125** | A4 對 A5_ucur | **是（全篇風險最高的一句）** |
| §4.3 :252 | **0.025**、**0.053** | 與 QPMIL-VL 的差距（由我方數字導出） | **是** |
| §4.3 :252 | 「within 0.01」、0.839、0.812 | 外部／零樣本 | 否 |
| §4.4 :257 | **−0.011**、**+0.008** | 移除 group 層 | **是** |
| §5 :266 | sd **0.03–0.05** | 折間變異「大於頂尖方法之間的差距」 | **是**（|M| 變小可能放大） |

⚠️ **摘要為老師所寫。** 上表中 Abstract 的三項若改數字，需經老師確認。

---

## 5. Table 1 的完整 protocol（DR-054 第三階段）

從 `outputs/exp3/runs/*/meta.json` 記錄的 args 字串取得，這是最可靠的來源：

```bash
# reverse，每折一個 run，F ∈ {1..10}，seed = fold
python scripts/run_exp2.py --arms A5 --order reverse --arch hier --fold F --seeds F \
    --tag sota_ucur --head accumulating --out-root outputs/exp3 --uold current

# forward（⚠️ 旗標是 --order main，不是 forward）
python scripts/run_exp2.py --arms A5 --order main --arch hier --fold F --seeds F \
    --tag sota_ucur --head accumulating --out-root outputs/exp3 --uold current
```

`B=8` 是預設值、**沒有顯式傳**；`--mem-capacity` 也**沒有傳**，因此走
`MEMORY_CAPACITY` 的 512。本批只加上 `--mem-capacity 64` 與新的 `--tag`／`--out-root`。

---

## 6. 檔名是否含 |M|

含。`run_exp2.py:599`：`suffix = f"_M{args.mem_capacity}" if args.mem_capacity else ""`，
只在**顯式傳入且非零**時加。既有的 512 產物沒有 `_M` 後綴，所以不會撞名。

但聚合報表 `EXP2.md` / `EXP2_hier.md` 的路徑只由 `--tag` 決定，**會互相覆蓋**。
本批因此每個設定一個獨立 tag，寫進 `outputs/exp5/`（規則 4）。

---

## 7. 依賴 fold-1 五 seed（|M|=512）的正文分析

| 位置 | 主張 | 需要重跑的 run |
|---|---|---|
| §4.2 :225 | 「完整目標比只有 replay 高 2.5 點，五個 seed 全數成立」 | A5 ucur ×5、A3 ×5 |
| §4.2 :247 | 洩漏 34.8（只有蒸餾） | B1 ×5 |
| §4.2 :247 | 78.9 與 77.0 | B2 ucur ×5、A3 ×5 |
| §4.2 :247 | 79.5（完整目標最高） | A5 ucur ×5 |
| §4.2 :247 | Jaccard 0.165 對 0.125 的**反轉** | A4 ×5、A5 ucur ×5 |
| §4.4 :257 | warm-start 無系統性增益（目前是定性敘述） | W1 ×5（僅在改為定量時） |

最小重跑集合（fold 1、seeds 0–4、flat、reverse、accumulating）：
B1、A3、A4 走 `--uold snapshot`（15 run）＋ B2、A5 走 `--uold current`（10 run）＝ **25 run**。
A2 可直接沿用。

---

## 8. 「5090 新數字」欄位

見下一節。

---

## 9. 5090 新數字（|M| = 64，本批實測）

格式與稿件相同：ACC ± sd / Forgetting / Masked。**所有數字從 `outputs/exp5/` 的輸出檔讀取。**

| 位置 | 舊數字（\|M\|=512） | 新數字（\|M\|=64） | 來源目錄 |
|---|---|---|---|
| Table 1 `:195` PathSelect reverse | .834±.031 / .101 / .914 | 0.755 ± 0.021 / 0.206 / 0.906 | `outputs/exp5/m64_full_rev` |
| Table 1 `:195` PathSelect forward | .837±.027 / .059 / .916 | 0.820 ± 0.033 / 0.091 / 0.914 | `outputs/exp5/m64_full_fwd` |
| Table 2 `:215` ＋replay reverse | .819±.042 / .121 | 0.757 ± 0.031 / 0.222 | `outputs/exp5/m64_replay_rev` |
| Table 2 `:215` ＋replay forward | .831±.044 / .078 | 0.823 ± 0.027 / 0.103 | `outputs/exp5/m64_replay_fwd` |
| Table 2 `:216` ＋蒸餾＋效用下限 reverse | .823±.041 / .108 | 0.762 ± 0.046 / 0.205 | `outputs/exp5/m64_distutil_rev` |
| Table 2 `:216` ＋蒸餾＋效用下限 forward | .846±.047 / .068 | 0.821 ± 0.038 / 0.099 | `outputs/exp5/m64_distutil_fwd` |
| Table 2 `:217` ＋group 層預算 reverse | .834±.031 / .101 | 0.755 ± 0.021 / 0.206 | `outputs/exp5/m64_full_rev` |
| Table 2 `:217` ＋group 層預算 forward | .837±.027 / .059 | 0.820 ± 0.033 / 0.091 | `outputs/exp5/m64_full_fwd` |
| 摘要 `:62`、§1 `:90` reverse 終點 | 0.834 | 0.755 | `outputs/exp5/m64_full_rev` |
| 摘要 `:62`、§1 `:90` forward 終點 | 0.837 | 0.820 | `outputs/exp5/m64_full_fwd` |
| §1 `:96`、§4.1 `:167` 記憶體容量 | 512 snapshots | **64 snapshots** | `設定值` |
| §4.2 `:225` replay 恢復 reverse | 0.819 | 0.757 | `outputs/exp5/m64_replay_rev` |
| §4.2 `:225` replay 恢復 forward | 0.831 | 0.823 | `outputs/exp5/m64_replay_fwd` |
| §4.2 `:225` 本方法 − replay 的增益 | +0.015（6/10）、+0.006（7/10） | **−0.0020（6/10）、−0.0027（4/10）** | `本批逐折配對` |
| §4.3 `:252` 與 QPMIL-VL 的差距 | 0.025 / 0.053 | **0.104 / 0.070**（.859−.755、.890−.820） | `由本批數字導出` |
| §4.4 `:257` 移除 group 層 | −0.011 / +0.008 | **+0.0068（5/10）／+0.0003（5/10）** | `本批逐折配對` |
| §5 `:266` 折間 sd | 0.03–0.05 | 0.021–0.046 | `本批六組` |

⚠️ 舊數字來自 `outputs/exp3`（**不同批次**）。本表只是並列，**不是配對比較**，兩欄不得相減。
⚠️ Table 3 與其相關的正文分析（fold 1 五 seed）**本批未重跑**，見 `docs/EXP5_REPORT.md`。

