#!/bin/zsh
# 四段式 pipeline（DR-031 排程）。憲法 §3.5（失敗語意）、§3.7（存活訊號）。
#
# ⚠️ zsh 不對未加引號的變數做 word splitting（bash 會）。
#    R="python foo.py"; $R  →  zsh 把整串當成一個檔名。
#    因此本檔**不使用變數當命令**，每一行都是完整可執行的命令。
set -euo pipefail
cd /Users/aaron/research/02_pathselect            # 絕對路徑，不依賴啟動時的 cwd

JOB=pipeline
TOTAL=4

heartbeat () {   # [heartbeat] <ISO時間> stage=<n/N> <描述>
  echo "[heartbeat] $(date -Iseconds) stage=$1/${TOTAL} $2"
  python scripts/job_status.py --job "${JOB}" --state running --stage "$1/${TOTAL}" --note "$2"
}
fail () {
  echo "[FAILED] stage=${CURRENT_STAGE:-?} $1"
  python scripts/job_status.py --job "${JOB}" --state failed \
      --stage "${CURRENT_STAGE:-?}/${TOTAL}" --note "$1" || true
}
trap 'fail "第 ${CURRENT_STAGE:-?} 段以非零狀態結束（見 log 上方）"' ERR

stamp () {
  echo "[freeze] $1 HEAD=$(git rev-parse --short HEAD)" \
       "run_exp2=$(git log -1 --format=%h -- scripts/run_exp2.py)" \
       "rounds=$(git log -1 --format=%h -- selector/rounds.py)"
}

while pgrep -f "run_exp2.py|b1_fill.sh" >/dev/null 2>&1; do sleep 60; done
stamp "pipeline 啟動"
CURRENT_STAGE=0
python scripts/job_status.py --job "${JOB}" --state running --stage "0/${TOTAL}" --note "啟動"

CURRENT_STAGE=1
heartbeat 1 "G1'-b：noGroupKD（階層版，5 seeds）"
python scripts/run_exp2.py --arms A5nG --order reverse --seeds 0,1,2,3,4 \
    --beta-u 0.1 --arch hier --allocation per_budget --tag hier2
python scripts/check_batch_products.py --tag hier2 --arms A5,A3,A5nG --seeds 0,1,2,3,4 --suffix _hier
python scripts/report_hier2.py
echo "########## 1/4 G1'-b 完成 ##########"

CURRENT_STAGE=2
heartbeat 2 "G2：L_sem 三臂（階層版，5 seeds）"
python scripts/run_exp2.py --arms A5 --order reverse --seeds 0,1,2,3,4 \
    --beta-u 0.1 --arch hier --allocation per_budget --prior none --tag prior
python scripts/run_exp2.py --arms A5 --order reverse --seeds 0,1,2,3,4 \
    --beta-u 0.1 --arch hier --allocation per_budget --prior max_sim --tag prior
python scripts/check_batch_products.py --tag prior --arms A5 --seeds 0,1,2,3,4 --suffix _hier_none
python scripts/check_batch_products.py --tag prior --arms A5 --seeds 0,1,2,3,4 --suffix _hier_max_sim
python scripts/report_prior.py hier
echo "########## 2/4 G2 完成 ##########"

CURRENT_STAGE=3
heartbeat 3 "main order 的 A3/A5 補到 5 seeds"
python scripts/run_exp2.py --arms A3,A5 --order main --seeds 0,1,2,3,4 --beta-u 0.1 --tag order_main
python scripts/check_batch_products.py --tag order_main --arms A3,A5 --seeds 0,1,2,3,4 --order main
python scripts/run_exp2.py --arms A1,A2,A3,A4,A5,R1,R2 --order main --seeds 0,1,2 \
    --tag order_main --report-only
python scripts/report_order_dependence.py
echo "########## 3/4 main order 補齊完成 ##########"

CURRENT_STAGE=4
heartbeat 4 "E1 階層版（方案 A，40 輪）"
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
python scripts/report_memory_hier.py
echo "########## 4/4 E1 階層版完成 ##########"

stamp "pipeline 結束"
python scripts/job_status.py --job "${JOB}" --state done --stage "${TOTAL}/${TOTAL}" --note "全部完成"
echo "########## PIPELINE 全部完成 ##########"
