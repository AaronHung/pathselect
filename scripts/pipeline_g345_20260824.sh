#!/bin/zsh
# G5 → G4 → G3（架構完整性）。等 pipeline_stage4（E1）退出後才開始。
# G345 與 E1 沒有資料相依（不同 tag），所以 E1 即使失敗也照跑，只記錄其最終狀態。
set -euo pipefail
cd /Users/aaron/research/02_pathselect
JOB=pipeline_g345
TOTAL=4
CURRENT_STAGE=0
fail () {
  echo "[FAILED] stage=${CURRENT_STAGE:-?} $1"
  python scripts/job_status.py --job "${JOB}" --state failed --stage "${CURRENT_STAGE}/${TOTAL}" --note "$1" || true
}
trap 'fail "第 ${CURRENT_STAGE} 段以非零狀態結束（見 log 上方）"' ERR

python scripts/job_status.py --job "${JOB}" --state running --stage "0/${TOTAL}" --note "等待 E1 結束"
echo "[wait] $(date -Iseconds) 等待 pipeline_stage4 結束…"
while pgrep -f pipeline_stage4_20260824.sh > /dev/null; do
  echo "[heartbeat] $(date -Iseconds) E1 仍在跑，memory_hier 產物 $(ls outputs/exp2/memory_hier/per_slide 2>/dev/null | wc -l | tr -d ' ')/40"
  sleep 600
done
echo "[wait] $(date -Iseconds) E1 已結束，最終狀態："
cat outputs/_status/pipeline_stage4.json || true

echo "[freeze] 啟動 HEAD=$(git rev-parse --short HEAD)" \
     "run_arch=$(git log -1 --format=%h -- scripts/run_arch_completeness.py)" \
     "sem_loss=$(git log -1 --format=%h -- selector/sem_loss.py)"
mkdir -p outputs/exp2/arch/per_slide

# ── stage 1：G5（state） ──
CURRENT_STAGE=1
echo "[heartbeat] $(date -Iseconds) stage=1/${TOTAL} G5 state 條件化（5 輪）"
python scripts/job_status.py --job "${JOB}" --state running --stage "1/${TOTAL}" --note "G5 state 5 輪"
python scripts/check_state_noop.py
python scripts/run_arch_completeness.py g5 --seeds 0,1,2,3,4
python scripts/check_batch_products.py --tag arch --arms A5 --seeds 0,1,2,3,4 --suffix _hier_state

# ── stage 2：G4（q_tau） ──
CURRENT_STAGE=2
echo "[heartbeat] $(date -Iseconds) stage=2/${TOTAL} G4 q_tau 條件化（5 輪）"
python scripts/job_status.py --job "${JOB}" --state running --stage "2/${TOTAL}" --note "G4 q_tau 5 輪"
python -m pytest tests/test_leakage.py -q
python scripts/run_arch_completeness.py g4 --seeds 0,1,2,3,4
python scripts/check_batch_products.py --tag arch --arms A5 --seeds 0,1,2,3,4 --suffix _hier_query

# ── stage 3：G3（group L_sem） ──
CURRENT_STAGE=3
echo "[heartbeat] $(date -Iseconds) stage=3/${TOTAL} G3 group L_sem beta_g=0.1（5 輪）"
python scripts/job_status.py --job "${JOB}" --state running --stage "3/${TOTAL}" --note "G3 group L_sem 5 輪"
python scripts/run_arch_completeness.py g3 --seeds 0,1,2,3,4 --beta-g 0.1
python scripts/check_batch_products.py --tag arch --arms A5g --seeds 0,1,2,3,4 --suffix _hier

# ── stage 4：報告 ──
CURRENT_STAGE=4
echo "[heartbeat] $(date -Iseconds) stage=4/${TOTAL} 產生 ARCH_COMPLETENESS.md"
python scripts/job_status.py --job "${JOB}" --state running --stage "4/${TOTAL}" --note "產生報告"
python scripts/report_arch_completeness.py
test -s outputs/exp2/arch/ARCH_COMPLETENESS.md
# 用 if 而非 `grep -q ... && { ... }`：後者可讀性差（實測 set -e 兩者都安全，
# 因為 AND-OR list 中非最後一個命令的失敗是豁免的）。
if grep -q "PENDING" outputs/exp2/arch/ARCH_COMPLETENESS.md; then
  fail "報告仍有 PENDING，代表有實驗沒跑完"
  exit 1
fi

echo "[freeze] 結束 HEAD=$(git rev-parse --short HEAD)"
python scripts/job_status.py --job "${JOB}" --state done --stage "${TOTAL}/${TOTAL}" --note "G5/G4/G3 完成，報告已產出"
echo "########## G345 完成（產物已檢查、報告已產出）##########"
