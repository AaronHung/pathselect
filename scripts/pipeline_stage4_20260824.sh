#!/bin/zsh
# 只跑 pipeline 的第 4 段（E1 階層版）。1-3 段已完成，不重跑。
set -euo pipefail
cd /Users/aaron/research/02_pathselect
JOB=pipeline_stage4
TOTAL=1
CURRENT_STAGE=1
fail () {
  echo "[FAILED] stage=${CURRENT_STAGE:-?} $1"
  python scripts/job_status.py --job "${JOB}" --state failed --stage "${CURRENT_STAGE}/${TOTAL}" --note "$1" || true
}
trap 'fail "E1 階層版以非零狀態結束（見 log 上方）"' ERR

echo "[freeze] 啟動 HEAD=$(git rev-parse --short HEAD)" \
     "run_exp2=$(git log -1 --format=%h -- scripts/run_exp2.py)" \
     "rounds=$(git log -1 --format=%h -- selector/rounds.py)"
echo "[heartbeat] $(date -Iseconds) stage=1/1 E1 階層版（方案 A，40 輪）"
python scripts/job_status.py --job "${JOB}" --state running --stage "1/${TOTAL}" --note "E1 階層版 40 輪"

mkdir -p outputs/exp2/memory_hier/per_slide
for A in A3 A5; do
  for S in 0 1 2 3 4; do
    cp -n "outputs/exp2/hier2/per_slide/${A}_reverse_seed${S}_hier.json" \
          "outputs/exp2/memory_hier/per_slide/${A}_reverse_seed${S}_M512_hier.json"
  done
done
for M in 64 128 256 1024; do
  echo "===== |M| = ${M} ====="
  python scripts/run_exp2.py --arms A3,A5 --order reverse --seeds 0,1,2,3,4 \
      --beta-u 0.1 --arch hier --allocation per_budget --mem-capacity "${M}" --tag memory_hier
  python scripts/check_batch_products.py --tag memory_hier --arms A3,A5 \
      --seeds 0,1,2,3,4 --suffix "_M${M}_hier"
done
python scripts/check_batch_products.py --tag memory_hier --arms A3,A5 --seeds 0,1,2,3,4 --suffix "_M512_hier"
python scripts/report_memory_hier.py
echo "[freeze] 結束 HEAD=$(git rev-parse --short HEAD)"
python scripts/job_status.py --job "${JOB}" --state done --stage "1/${TOTAL}" --note "E1 階層版完成"
echo "########## E1 階層版完成（產物已檢查、報告已產出）##########"
