#!/bin/bash
gpu_id="1"
dataset="Scientific"
ckp=150
# 解析命令行参数
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --gpu_id)
            gpu_id="$2"
            shift
            shift
            ;;
        --dataset)
            dataset="$2"
            shift
            shift
            ;;
        --ckp)
            ckp="$2"
            shift
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

output_file="sh_result/${dataset}_lambda_res_ckp${ckp}.txt"

# before 2024.4.7
# for lambda1 in 0.1 0.2 0.5 1 1.5 3; do
#     for lambda2 in  0.1 0.2 0.5 1 1.5 3; do
#         python run_test.py --Ours --gpu_id $gpu_id --ckp $ckp --data_name $dataset --lambda1 $lambda1 --lambda2 $lambda2 >> $output_file
#         echo "lambda1: $lambda1, lambda2: $lambda2" >> $output_file
#         echo "" >> $output_file
#     done
# done

for lambda1 in 0.12 0.15 0.17 0.19; do
    for lambda2 in  0.05 0.08 0.1 0.12 0.15 0.17 0.19; do
        python run_test.py --Ours --gpu_id $gpu_id --ckp $ckp --data_name $dataset --lambda1 $lambda1 --lambda2 $lambda2 >> $output_file
        echo "lambda1: $lambda1, lambda2: $lambda2" >> $output_file
        echo "" >> $output_file
    done
done
