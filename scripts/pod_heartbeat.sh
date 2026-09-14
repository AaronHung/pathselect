#!/usr/bin/env bash
# DR-052 pod 監看器：每 10 分鐘把進度追加到 logs/pod/heartbeat.log。
#   nohup bash scripts/pod_heartbeat.sh > /dev/null 2>&1 &
# 欄位：時間｜已完成 run 數/總數（所有 batch*.runs 的聯集）｜執行中的 run｜最近一筆 class-IL｜
#       GPU 使用率（訓練路徑是 CPU，此欄預期為 0）｜負載｜剩餘磁碟。
set -u
cd "$(dirname "$0")/.."
mkdir -p logs/pod
INTERVAL="${1:-600}"
while true; do
  total=0; done_=0
  for L in logs/pod/batch*.runs; do
    [ -f "$L" ] || continue
    for n in $(cut -d'|' -f1 "$L"); do
      total=$((total + 1)); [ -f "outputs/exp3/runs/$n/DONE" ] && done_=$((done_ + 1))
    done
  done
  running=$(pgrep -af "run_exp2.py" | grep -v pgrep | sed -n 's/.*--arms \([^ ]*\) --order \([^ ]*\) --arch \([^ ]*\) --fold \([0-9]*\) --seeds \([0-9]*\).*/\1-\3-\2-f\4s\5/p' | tr '\n' ',' | sed 's/,$//')
  last=$(ls -t outputs/exp3/runs/*/run.log 2>/dev/null | head -1)
  lastcl=$( [ -n "$last" ] && grep "class-IL=" "$last" | tail -1 | sed 's/^ *//' || echo "-")
  gpu=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 || echo "n/a")
  load=$(cut -d' ' -f1 /proc/loadavg)
  disk=$(df -h /workspace | awk 'NR==2{print $4}')
  echo "[$(date '+%F %T')] done=${done_}/${total} running=[${running:-none}] last='${lastcl}' gpu=${gpu}% load=${load} free=${disk}" >> logs/pod/heartbeat.log
  sleep "$INTERVAL"
done
