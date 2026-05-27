#!/bin/bash
# ============================================================================
# Multi-Seed Evaluation Script
# Usage: bash run_multi_seed.sh --data_name Scientific --Ours --MMOE
# ============================================================================
set -e

SEEDS=(23 42 101 3407 9999)
DATASET="${DATASET:-Scientific}"
GPU="${GPU:-0}"
ARGS="$@"

OUTPUT="output_multi_seed/${DATASET}_multi_seed.txt"
mkdir -p output_multi_seed

echo "Multi-Seed Evaluation on ${DATASET} with seeds: ${SEEDS[*]}"
echo "Results saved to: ${OUTPUT}"
echo "==============================================" | tee "$OUTPUT"

for seed in "${SEEDS[@]}"; do
    echo ">>> Seed = $seed" | tee -a "$OUTPUT"
    python run_test.py \
        --output_dir "output_multi_seed/seed_${seed}/" \
        --data_name "$DATASET" \
        --gpu_id "$GPU" \
        --seed $seed \
        $ARGS \
        --epochs 250 \
        --patience 999 \
        2>&1 | grep -E "HIT@10|NDCG@10|full metric" | tee -a "$OUTPUT"
    echo "" | tee -a "$OUTPUT"
done

echo "==============================================" | tee -a "$OUTPUT"
echo "Done. Full results in: $OUTPUT"
