#!/bin/bash
# ============================================================================
# FCL-MSR Experiment Script
# Usage:
#   bash run.sh ablation  --dataset Scientific    # Ablation study
#   bash run.sh lambda    --dataset Scientific    # Lambda grid search
#   bash run.sh mmoe      --dataset Scientific    # Expert count search
#   bash run.sh baseline                         # Table2 baseline comparison
# ============================================================================

set -e

MODE="${1:-help}"
DATASET="${DATASET:-Scientific}"
GPU_ID="${GPU_ID:-0}"
CKP="${CKP:-200}"

usage() {
    echo "Usage: bash run.sh <mode> [options]"
    echo ""
    echo "Modes:"
    echo "  ablation   Ablation study (ablation_code 2~6)"
    echo "  lambda     Lambda1/Lambda2 modality weight grid search"
    echo "  mmoe       Expert count grid search"
    echo "  baseline   Table2 baseline comparison (baseline/run_experiment.py)"
    echo ""
    echo "Environment variables:"
    echo "  DATASET    Dataset name (default: Scientific)"
    echo "  GPU_ID     GPU ID (default: 0)"
    echo "  CKP        Pretrain checkpoint epochs (default: 200)"
    exit 0
}

case "$MODE" in
    help|--help|-h)
        usage
        ;;
    
    ablation)
        # Ablation: CLIP feat(2), Attr encoder(3), MOE(4), no text(5), no image(6)
        case "$DATASET" in
            Scientific) L1=0.2; L2=0.1; MAIN=6; MODAL=6 ;;
            Pantry)     L1=1.0; L2=0.5; MAIN=4; MODAL=2 ;;
            Arts)       L1=0.2; L2=0.1; MAIN=4; MODAL=8 ;;
            Instruments)L1=0.5; L2=0.1; MAIN=10; MODAL=8 ;;
            *)          L1=0.5; L2=0.5; MAIN=4; MODAL=4 ;;
        esac
        OUTPUT="output/${DATASET}_ablation.txt"
        mkdir -p output
        for i in 2 3 4 5 6; do
            echo "=== ablation_code=$i ==="
            python run_test.py --data_name "$DATASET" --ablation_code $i --Ours \
                --ckp "$CKP" --lambda1 "$L1" --lambda2 "$L2" \
                --main_expert_num "$MAIN" --modal_expert_num "$MODAL" \
                --gpu_id "$GPU_ID" | tee -a "$OUTPUT"
        done
        ;;
    
    lambda)
        # Lambda1/Lambda2 modality weight grid search
        OUTPUT="output/${DATASET}_lambda_search.txt"
        mkdir -p output
        for lambda1 in 0.1 0.2 0.5 1.0 1.5 3.0; do
            for lambda2 in 0.1 0.2 0.5 1.0 1.5 3.0; do
                echo "=== lambda1=$lambda1 lambda2=$lambda2 ==="
                python run_test.py --Ours --gpu_id "$GPU_ID" --ckp "$CKP" \
                    --data_name "$DATASET" --lambda1 "$lambda1" --lambda2 "$lambda2" \
                    | tee -a "$OUTPUT"
            done
        done
        ;;
    
    mmoe)
        # Expert count grid search
        L1=1; L2=3  # Default lambda values
        OUTPUT="output/${DATASET}_mmoe_search.txt"
        mkdir -p output
        for main_n in 10 8 6 4 2; do
            for modal_n in 10 8 6 4 2; do
                echo "=== main_expert=$main_n modal_expert=$modal_n ==="
                python run_test.py --Ours --gpu_id "$GPU_ID" --ckp "$CKP" \
                    --lambda1 "$L1" --lambda2 "$L2" --data_name "$DATASET" \
                    --main_expert_num "$main_n" --modal_expert_num "$modal_n" \
                    | tee -a "$OUTPUT"
            done
        done
        ;;
    
    baseline)
        # Table2 baseline comparison
        for ds in LastFM Beauty Toys_and_Games Sports_and_Outdoors Yelp; do
            echo "=== Baseline: $ds ==="
            python baseline/run_experiment.py --data_name "$ds" --gpu_id "$GPU_ID"
        done
        ;;
    
    *)
        echo "Unknown mode: $MODE"
        usage
        ;;
esac
