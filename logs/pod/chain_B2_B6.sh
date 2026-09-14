#!/usr/bin/env bash
# 等 B1 的執行器結束，再依序跑 B2..B6（各自 nohup 語意由本鏈保證；每批寫 BATCH_DONE）
cd /workspace/pathselect
source /workspace/venvs/ps3/bin/activate
while pgrep -f "pod_run_exp3.sh 1 " > /dev/null; do sleep 60; done
echo "[$(date "+%F %T")] chain: B1 runner 已結束，開始 B2..B6" >> logs/pod/chain.log
for b in 2 3 4 5 6; do
  echo "[$(date "+%F %T")] chain: ▶ batch $b" >> logs/pod/chain.log
  bash scripts/pod_run_exp3.sh "$b" 10 8 > "logs/pod/batch$b.log" 2>&1
  echo "[$(date "+%F %T")] chain: ✔ batch $b 執行器結束（$(tail -1 logs/pod/batch$b.log)）" >> logs/pod/chain.log
done
echo "[$(date "+%F %T")] chain: ALL_DONE" >> logs/pod/chain.log
