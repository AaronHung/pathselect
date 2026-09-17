#!/usr/bin/env bash
# DR-056 第 7 步 **第二批**（batch 2）：四個 run 重跑，同批同機、四路平行。
#
# 為什麼重跑：第一批的 run 2／run 4 共用 --tag，因而共用側倉目錄、互相覆寫候選
# 特徵（證據見 commit 2cfc413 與 DR-056）。修法有二：側倉一個 run 一個目錄、
# 側倉檔加存 cand_idx 當身分憑據（對不上就硬失敗）。
#
# 第一批的 run 1／run 3（讀整張）不受影響，但仍一起重跑，理由有二：
#   (1) 四個 run 必須在同一批、同一台、同樣的競爭條件下，wall-clock 才可比；
#   (2) 兩批的 rfull 結果應逐位元相同，正好當作同機決定性的免費核對。
#
# 掛在第一批鏈尾：等到沒有 run_exp2.py 在跑才啟動，不與第一批並行。
set -u
cd /workspace/pathselect
source /workspace/venvs/ps4/bin/activate
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PATHSELECT_TORCH_THREADS=8
mkdir -p logs/exp4/b2

while [ "$(ps aux | grep -c '[r]un_exp2.py')" -gt 0 ]; do sleep 60; done
date +%s > logs/exp4/b2/BATCH.start
echo "第一批已清空，啟動 batch 2 $(date -Is)"

COMMON="--arms A5 --seeds 1 --fold 1 --arch hier --head accumulating --uold current \
--mem-capacity 128 --out-root outputs/exp4"

launch () {                       # launch <name> <order> <tag> [extra flags]
  local name=$1 order=$2 tag=$3; shift 3
  ( date +%s > logs/exp4/b2/${name}.start
    python scripts/run_exp2.py $COMMON --order "$order" --tag "$tag" "$@" \
      > logs/exp4/b2/${name}.log 2>&1
    echo $? > logs/exp4/b2/${name}.rc
    date +%s > logs/exp4/b2/${name}.end ) &
}

launch run1_rfull_rev  reverse cand_m128_rfull_b2
launch run2_r256_rev   reverse cand_m128_r256_b2  --replay-candidate-only
launch run3_rfull_fwd  main    cand_m128_rfull_b2
launch run4_r256_fwd   main    cand_m128_r256_b2  --replay-candidate-only

wait
date +%s > logs/exp4/b2/BATCH.end
for f in logs/exp4/b2/run*.rc; do echo "$f -> $(cat "$f")"; done
echo "BATCH2 DONE"
