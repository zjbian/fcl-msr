#!/bin/bash
gpu_id="1"
dataset="Scientific"
lambda1=1
lambda2=3
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
        --lambda1)
            lambda1="$2"
            shift
            shift
            ;;
        --lambda2)
            lambda2="$2"
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

lambda1=$(awk "BEGIN {print $lambda1+0; exit}")
lambda2=$(awk "BEGIN {print $lambda2+0; exit}")

output_file1="sh_result/${dataset}_mmoe_res.txt"
for i in 10 8 6 4 2; do
    for j in  10 8 6 4 2; do
        python run_test.py --Ours --gpu_id $gpu_id --ckp $ckp --lambda1 $lambda1 --lambda2 $lambda2 --data_name $dataset --main_expert_num $i --modal_expert_num $j >> $output_file1
        echo "main_expert_num: $i, modal_expert_num: $j" >> $output_file1
        echo "" >> $output_file1
    done
done