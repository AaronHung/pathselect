# GRAVEYARD（L3 墓園 —— 柴火區）

每條都有復活條件。被砍不等於被否證；`PARKED` 的東西是下一篇論文的種子。

| G | 被砍的方向 | 砍它的證據 | 復活條件（= 下一篇論文的種子） |
|---|---|---|---|
| G-01 | QPMIL（方法） | 老師否決；殘留已於 DR-002/003 清除 | 無 |
| G-02 | MoE experts / mechanism probing | navipath ADR-0006（前朝已砍） | 無 |
| G-03 | MLLM-HWSI 多尺度特徵管線 | 老師叫停（抽取成本） | 若未來有預算重抽：多尺度 coarse view 直接解 G-07 |
| G-04 | PathAgent 完整實作 | 前處理比被叫停的還重；AdaptivePath 已佔位 | 寫 agent-loop 整合論文時，以 scaffold 身分復活 |
| G-05 | **task-conditioning / q_τ 主張** | S1：跨器官任務 98.2/98.6% 線性可分 → q 結構性冗餘 | **同器官多任務標籤**（BRCA subtype/grade/receptor——老師原始任務序列）。這是最肥的一根柴 |
| G-06 | FiLM 調變救 q_τ | 同 G-05：可分離時 FiLM 只是強迫使用冗餘輸入 | 僅當新 benchmark 的 separability probe 顯著 <100% |
| G-07 | partial observation / "Navigation" 用字 | g_j 由同批 patch feature 算出，nothing hidden | 獨立 coarse 表徵（縮圖 encoder 或 G-03 復活） |
| G-08 | per-task LoRA bank 當方法 | 推論需 task id、儲存線性成長、零遷移 | 永久降級為 specialist reference（已在論文中有位置） |
| G-09 | L5 hierarchy / L6 stateful 進正文 | 8/22 調度：SeqFT 落差成主軸，L5/L6 讓位 | 時間允許→附錄 component ablation；否則→下一篇的主實驗（含 per-round L_util 修正） |
| G-10 | mllm can_dataset 當實驗 cohort | 每 task test 僅 6 張，一張=16.7pp | 僅作規格書/概念來源，永不當主表 |
| G-11 | 牌二：Selection Memory 多樣性取樣 | 未被否證，僅排序在牌一之後 | E1 後若仍需第二個機制貢獻，隨時可打 |
