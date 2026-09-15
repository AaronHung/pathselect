#!/usr/bin/env bash
# B7（Prompt 22）：等 B2..B6 鏈結束（chain.log 出現 ALL_DONE、無 pod_run_exp3.sh 在跑、B6 的 30 個 DONE 齊全）
# 才依序跑 batch 7（反向 A1/A3）與 8（正向 A1/A3）。絕不與 B6 並行。
cd /workspace/pathselect
source /workspace/venvs/ps3/bin/activate
b6_done() { n=0; for r in $(cut -d"|" -f1 logs/pod/batch6.runs 2>/dev/null); do [ -f "outputs/exp3/runs/$r/DONE" ] && n=$((n+1)); done; [ "$n" -eq 30 ]; }
while ! grep -q "ALL_DONE" logs/pod/chain.log 2>/dev/null || pgrep -f "pod_run_exp3.sh" > /dev/null || ! b6_done; do sleep 60; done
echo "[$(date "+%F %T")] chain_B7: B6 全部 DONE，開始 B7-1（batch 7）" >> logs/pod/chain.log
for b in 7 8; do
  echo "[$(date "+%F %T")] chain_B7: ▶ batch $b" >> logs/pod/chain.log
  bash scripts/pod_run_exp3.sh "$b" 10 8 > "logs/pod/batch$b.log" 2>&1
  echo "[$(date "+%F %T")] chain_B7: ✔ batch $b 執行器結束（$(tail -1 logs/pod/batch$b.log)）" >> logs/pod/chain.log
done
echo "[$(date "+%F %T")] chain_B7: B7_ALL_DONE" >> logs/pod/chain.log
