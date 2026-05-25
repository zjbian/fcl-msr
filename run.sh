#!/bin/bash
# ============================================================================
# FCL-MSR 实验脚本
# 用法:
#   bash run.sh ablation  --dataset Scientific    # 消融实验
#   bash run.sh lambda    --dataset Scientific    # λ1/λ2 网格搜索
#   bash run.sh mmoe      --dataset Scientific    # 专家数量搜索
#   bash run.sh baseline                         # Table2 基线对比
# ============================================================================

set -e

MODE="${1:-help}"
DATASET="${DATASET:-Scientific}"
GPU_ID="${GPU_ID:-0}"
CKP="${CKP:-200}"

usage() {
    echo "用法: bash run.sh <mode> [options]"
    echo ""
    echo "模式:"
    echo "  ablation   消融实验 (ablation_code 2~6)"
    echo "  lambda     λ1/λ2 模态权重网格搜索"
    echo "  mmoe       专家数量网格搜索"
    echo "  baseline   Table2 基线模型对比 (run_experiment.py)"
    echo ""
    echo "环境变量:"
    echo "  DATASET    数据集名称 (默认: Scientific)"
    echo "  GPU_ID     GPU ID (默认: 0)"
    echo "  CKP        预训练轮数 (默认: 200)"
    exit 0
}

case "$MODE" in
    help|--help|-h)
        usage
        ;;
    
    ablation)
        # 消融实验: clip特征提取(2), 属性编码(3), MOE(4), 去除文本(5), 去除图像(6)
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
        # λ1/λ2 模态权重网格搜索
        OUTPUT="output/${DATASET}_lambda_search.txt"
        mkdir -p output
        for lambda1 in 0.1 0.2 0.5 1.0 1.5 3.0; do
            for lambda2 in 0.1 0.2 0.5 1.0 1.5 3.0; do
                echo "=== λ1=$lambda1 λ2=$lambda2 ==="
                python run_test.py --Ours --gpu_id "$GPU_ID" --ckp "$CKP" \
                    --data_name "$DATASET" --lambda1 "$lambda1" --lambda2 "$lambda2" \
                    | tee -a "$OUTPUT"
            done
        done
        ;;
    
    mmoe)
        # 专家数量网格搜索
        L1=1; L2=3  # 默认 λ 值
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
        # Table2 基线对比
        for ds in LastFM Beauty Toys_and_Games Sports_and_Outdoors Yelp; do
            echo "=== Baseline: $ds ==="
            python run_experiment.py --data_name "$ds" --gpu_id "$GPU_ID"
        done
        ;;
    
    *)
        echo "未知模式: $MODE"
        usage
        ;;
esac
