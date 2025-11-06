#!/bin/bash

dataset="Scientific"
output_file1="sh_ablation/${dataset}_res.txt"

# ablation_code:
#   1: for test
#   2: for clip extract feature
#   3: for attr encoder
#   4: for moe Done
#   5: for modal fusion without text emb
#   6: for modal fusion without img emb


#Scientific

# for i in 2 3 4 5 6; do
#     python run_test.py --ablation_code $i --Ours --ckp 150 --lambda1 0.2 --lambda2 0.1 --main_expert_num 6 --modal_expert_num 6 >> $output_file1
# done

# dataset="Pantry"
# output_file1="sh_ablation/${dataset}_res.txt"
# for i in 2 3 4 5 6; do
#     python run_test.py --data_name $dataset --ablation_code $i --Ours --ckp 200 --lambda1 1.0 --lambda2 0.5 --main_expert_num 4 --modal_expert_num 2 >> $output_file1
# done

# dataset="Arts"
# output_file1="sh_ablation/${dataset}_res.txt"
# for i in 2 3 4 5 6; do
#     python run_test.py --data_name $dataset --ablation_code $i --Ours --ckp 200 --lambda1 0.2 --lambda2 0.1 --main_expert_num 4 --modal_expert_num 8 >> $output_file1
# done

dataset="Instruments"
output_file1="sh_ablation/${dataset}_res.txt"
for i in 2 3 4 5 6; do
    python run_test.py --data_name $dataset --ablation_code $i --Ours --ckp 200 --lambda1 0.5 --lambda2 0.1 --main_expert_num 10 --modal_expert_num 8 >> $output_file1
done