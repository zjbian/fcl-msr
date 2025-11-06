# -*- coding: utf-8 -*-
# @Time    : 2020/4/25 22:59
# @Author  : Hui Wang

import os
import numpy as np
import random
import torch
import argparse
import pickle

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from new_dataset import SASRecDataset
from new_trainers import FinetuneTrainer
from new_models import S3RecModel
from utils import EarlyStopping, get_user_seqs, get_item2attribute_json, check_path, set_seed, data_aug

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', default='../../sequential_rec/data/', type=str)
    parser.add_argument('--output_dir', default='output/tests1/bzj_2K_', type=str)
    parser.add_argument('--data_name', default='Instruments', type=str)

    parser.add_argument('--do_eval', action='store_true')
    parser.add_argument('--ckp', default=0, type=int, help="pretrain epochs 10, 20, 30...")

    # model args
    parser.add_argument("--model_name", default='SASRec', type=str)
    parser.add_argument("--hidden_size", type=int, default=32, help="hidden size of transformer model")
    parser.add_argument("--num_hidden_layers", type=int, default=2, help="number of layers")
    parser.add_argument('--num_attention_heads', default=2, type=int)
    parser.add_argument('--hidden_act', default="gelu", type=str) # gelu relu
    parser.add_argument("--attention_probs_dropout_prob", type=float, default=0.5, help="attention dropout p")
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3, help="hidden dropout p")
    parser.add_argument("--initializer_range", type=float, default=0.01)
    parser.add_argument('--max_seq_length', default=50, type=int)

    # train args
    parser.add_argument("--lr", type=float, default=0.002, help="learning rate of adam")
    parser.add_argument("--batch_size", type=int, default=256, help="number of batch_size")

    parser.add_argument("--epochs", type=int, default=150, help="number of epochs")

    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--log_freq", type=int, default=1, help="per epoch print res")
    parser.add_argument("--seed", default=26, type=int)

    parser.add_argument("--weight_decay", type=float, default=0.0, help="weight_decay of adam")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam second beta value")
    parser.add_argument("--gpu_id", type=str, default="0", help="gpu_id")

    parser.add_argument("--patience", type=int, default=8, help="patience for early stop")

    # our add params
    parser.add_argument("--sample_num", type=int, default=0, help="sample num of rank loss ")
    parser.add_argument("--aggregation", type=str, default="mean", help="loss aggregation way")
    parser.add_argument("--rank_act", type=str, default="softmax", help="rank similarith act")
    parser.add_argument("--isfull", type=int, default=0, help="TODO")
    ##MAP
    parser.add_argument("--MMOE", type=str, default='True', help="TODO")
    parser.add_argument("--attr_task", type=str, default='True', help="TODO")
    parser.add_argument("--MLP", type=str, default='False', help="TODO")
    parser.add_argument("--Trans", type=str, default='False', help="TODO")
    ## loss function
    parser.add_argument("--loss_type" , type=str, default='attr_loss', help="TODO")

    parser.add_argument("--mask_prod", type=float, default=0.2, help="TODO")
    parser.add_argument("--step", type=int, default=1, help="GNN Cell step")
    parser.add_argument("--Ours", action="store_true")
    parser.add_argument("--no_single_img", action="store_true")
    parser.add_argument("--lambda1", type=float, default=1, help="TODO")
    parser.add_argument("--lambda2", type=float, default=3, help="TODO")
    parser.add_argument("--dropout", type=float, default=0.4, help="TODO")
    parser.add_argument("--debug_index", type=float, default=0, help="TODO")
    parser.add_argument("--debug_dataset", action="store_true")
    parser.add_argument("--temperature", type=float, default=.0)
    parser.add_argument("--main_expert_num", type=int, default=4)
    parser.add_argument("--modal_expert_num", type=int, default=10)
    parser.add_argument("--num_heads", type=int, default=12)
    parser.add_argument("--ablation_code", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    args.output_dir += f'{args.data_name}/'
    check_path(args.output_dir)


    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    args.cuda_condition = torch.cuda.is_available() and not args.no_cuda
    
####数据加载
    args.data_file = args.data_dir + args.data_name + '.txt'
    item2attribute_file = args.data_dir + args.data_name + '_item2attributes.json'

    user_seq, max_item, valid_rating_matrix, test_rating_matrix = \
        get_user_seqs(args.data_file)

    item2attribute, attribute_size = get_item2attribute_json(item2attribute_file)

    args.item_size = max_item + 2
    args.mask_id = max_item + 1
    args.attribute_size = attribute_size + 1

    # if args.data_name == 'Pantry':
    #     args.item_size += 1


    # save model args
    args_str = f'{args.model_name}-{args.data_name}-{args.ckp}'
    args.log_file = os.path.join(args.output_dir, args_str + '.txt')
    print("________________LOG_FILE________________")
    print(args.log_file)

    # check_path(args.output_dir + f'{args.data_name}/')
    
    print("*****************")
    print(str(args))
    with open(args.log_file, 'a') as f:
        f.write(str(args) + '\n')

    args.item2attribute = item2attribute
    # set item score in train set to `0` in validation
    args.train_matrix = valid_rating_matrix

    # tensorboard 
    args.writer = None

    # 加载多模态的数据multi-modal
    print("*********args.Ours**********",args.Ours)
    if args.Ours:
        if args.no_single_img:        
            args.multi_modal_weight = pickle.load(open(f'/mnt/HDD2/lfy/SR_repo/multi_modal_dataset/{args.data_name}/{args.data_name}_clip_weights_finetune_minloss.pkl','rb'))
        else:
            args.multi_modal_weight = pickle.load(open(f'/mnt/HDD2/lfy/SR_repo/multi_modal_dataset/{args.data_name}/{args.data_name}_clip_weights_finetune_minloss_single.pkl','rb'))
        if args.ablation_code == 4:            
            args.MMOE = False
        else:
            args.MMOE = True
        args.MMOE = True
        args.MLP = False
        print(args.MMOE)
        args.attr_task = True

    args.clip_hidden_unit = 512
    # args.info = pickle.load(open('./data/newdata/new_beauty_info_finetune.pkl','rb'))
    args.multi_modal_update = True
    args.debug_code = 0
    # args.lambda1 = 1
    # args.lambda2 = 3

    # 1 1 1130
    # 0 2 1140
    # 0 3 1147 


    if hasattr(args, 'attr_task'):
        item2attr = {}
        for k,v in args.item2attribute.items():
            item2attr[int(k)] = v
        args.item2attr = item2attr
    # save model path
    checkpoint = args_str + '.pt'
    args.checkpoint_path = os.path.join(args.output_dir, checkpoint)


    if args.debug_dataset:
        train_dataset = SASRecDataset(args, data_aug(user_seq), data_type='train')
    else:
        train_dataset = SASRecDataset(args, user_seq, data_type='train')
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size)

    eval_dataset = SASRecDataset(args, user_seq, data_type='valid')
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.batch_size)

    test_dataset = SASRecDataset(args, user_seq, data_type='test')
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.batch_size)


    dataset2ckp = {'LastFM': 150,
                   'Beauty': 150,
                   'Sports_and_Outdoors': 100,
                   'Toys_and_Games': 150,
                   'Yelp': 100,
                  }

    def args_change(args, loss_type, model_name, hidden_size = None, code = None, update = None):
        args.loss_type = loss_type
        args.model_name = model_name
        args.train_matrix = valid_rating_matrix
        
        if hidden_size is not None:
            args.hidden_size = hidden_size

        if code is not None:
            args.debug_code = code

        if update is not None:
            args.multi_modal_update = update
            
        ### ours
        if model_name == 'S3Rec':
            print()
            args.ckp = dataset2ckp[args.data_name]

    model_args = {
                #   'Ori-SASRec-16': lambda: args_change(args, 'BCE', 'SASRec', 16),
                #   'Ori-SASRec-32': lambda: args_change(args, 'BCE', 'SASRec', 32),
                #   'Ori-SASRec-64': lambda: args_change(args, 'BCE', 'SASRec', 64),
                #   'Ori-SASRec-128': lambda: args_change(args, 'BCE', 'SASRec', 128),
                #   'Ori-SASRec-256': lambda: args_change(args, 'BCE', 'SASRec', 256),
                #   'Ori-SASRec-512-0-False': lambda: args_change(args, 'BCE', 'SASRec', 512, 0, False),
                #   'Ori-SASRec-512-0-True': lambda: args_change(args, 'BCE', 'SASRec', 512, 0, True),
                #   'Ori-SASRec-512-1-False': lambda: args_change(args, 'BCE', 'SASRec', 512, 1, False),
                #   'Ori-SASRec-512-1-True': lambda: args_change(args, 'BCE', 'SASRec', 512, 1, True),
                #   'Our-SASRec-16': lambda: args_change(args, 'CCE', 'SASRec', 16),
                #   'Our-SASRec-32': lambda: args_change(args, 'CCE', 'SASRec', 32),
                #   'Our-SASRec-64': lambda: args_change(args, 'CCE', 'SASRec', 64),
                #   'Our-SASRec-128': lambda: args_change(args, 'CCE', 'SASRec', 128),
                #   'Our-SASRec-256': lambda: args_change(args, 'CCE', 'SASRec', 256),
                #   'Our-SASRec-512-0-False': lambda: args_change(args, 'CCE', 'SASRec', 512, 1, False),
                #   'Our-SASRec-512-0-True': lambda: args_change(args, 'CCE', 'SASRec', 512, 1, True),
                 'Our-SASRec-512-1-False': lambda: args_change(args, 'CCE', 'SASRec', 64, 1, False),
                #   'Our-SASRec-512-1-False': lambda: args_change(args, 'CCE', 'SASRec', 64, 1, False),
                #   'Our-SASRec-512-1-True': lambda: args_change(args, 'CCE', 'SASRec', 64, 1, True),
                #   'Our-SASRec-512-0-False': lambda: args_change(args, 'CCE', 'SASRec', 1024, 0, False),
                #   'Our-SASRec-512-0-True': lambda: args_change(args, 'CCE', 'SASRec', 1024, 0, True),
                #   'Our-SASRec-512-1-False': lambda: args_change(args, 'CCE', 'SASRec', 1024, 1, False),
                #   'Our-SASRec-512-1-True': lambda: args_change(args, 'CCE', 'SASRec', 1024, 1, True),
                 }

    # model_args = {
    #               'Ori-GRU4Rec': lambda: args_change(args, 'CE', 'GRU4Rec'), 
    #               'Our-GRU4Rec': lambda: args_change(args, 'CCE', 'GRU4Rec'), 
    #               'Ori-SRGNN': lambda: args_change(args, 'CE', 'Ori-SRGNN'),
    #               'Our-SRGNN': lambda: args_change(args, 'CCE', 'SRGNN'),
    #               'Ori-SASRec': lambda: args_change(args, 'BCE', 'SASRec'), 
    #               'Our-SASRec': lambda: args_change(args, 'CCE', 'SASRec'), 
    #             #   'Ori-S3Rec': lambda: args_change(args, 'BCE', 'S3Rec'), 
    #             #   'Our-S3Rec': lambda: args_change(args, 'CCE', 'S3Rec'), 
    #              }

    model_result = {}

    for model_name in model_args:
        model_args[model_name]()
        

        args_str = f'{args.model_name}-{args.data_name}-{args.ckp}-{args.loss_type}'
        args.log_file = os.path.join(args.output_dir, args_str + '.txt')
        print

        with open(args.log_file, 'a') as f:
            f.write(str(args) + '\n')

        checkpoint = args_str + '.pt'
        args.checkpoint_path = os.path.join(args.output_dir, checkpoint)

        if 'GRU4Rec' == args.model_name:
            model = GRU4Rec(args=args)
        elif 'SRGNN' in args.model_name:
            model = SRGNN(args = args)
        ## ours model
        else:
            print("-------------model_name for train is S3RecModel----------------")
            model = S3RecModel(args=args)

        trainer = FinetuneTrainer(model, train_dataloader, eval_dataloader,
                                test_dataloader, args)

        if args.do_eval:
            
            #change pretrained_path from model saved last time 
            #pretrained_path = os.path.join(args.output_dir,args_str, +'.pt')
            
            
            pretrained_path = os.path.join('reproduce/', f'{args.data_name}-epochs-{args.ckp}.pt')
            # trainer.load(args.checkpoint_path)
            trainer.load(pretrained_path)
            # print(f'Load model from {args.checkpoint_path} for test!')
            print(f'Load model from {pretrained_path} for test!')
            scores, result_info = trainer.test(0, full_sort=True)
            print('Test result score:',scores)
            print('Test result:',result_info)    

        ### model_train
        else:
            print("________________pretained_path________________")
            pretrained_path = os.path.join('reproduce/', f'{args.data_name}-epochs-{args.ckp}.pt')
            print(pretrained_path)
            try:
                trainer.load(pretrained_path)
                print(f'Load Checkpoint From {pretrained_path}!')

            except FileNotFoundError:
                print(f'{pretrained_path} Not Found! The Model is same as SASRec')
            
            ##早停--10步之内无变化
            early_stopping = EarlyStopping(args.checkpoint_path, patience=args.patience,verbose=True)
            
            ##默认200轮
            for epoch in range(args.epochs):
                post_fix = trainer.train(epoch)

                scores, _ = trainer.valid(epoch, full_sort=True, verbose=True)
                print('Valid result score:',scores)
                #print('Valid result:',np.array(scores[-2:]))
                early_stopping(np.array(scores[-2:]), trainer.model)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break
            
            trainer.args.train_matrix = test_rating_matrix###？？？？？？？？？？？？？

            print('---------------Change to test_rating_matrix!-------------------')

            # load the best model
            print("our best model:", args.checkpoint_path)
            trainer.model.load_state_dict(torch.load(args.checkpoint_path))
            _, full_info = trainer.test(epoch, full_sort=True, verbose=True)
        
        print('------------------End of Test------------------')
        print(args_str)
        print('full metric:', full_info)
    
        with open(args.log_file, 'a') as f:
            f.write(args_str + '\n')
            # f.write(sample_info + '\n')
            f.write(full_info + '\n')
    
        model_result[model_name] = model_name + ' full metric: ' + str(full_info)  + '\n' + \
                                str(post_fix)
    
        del(model)
        del(trainer)
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()

        final_result_file = os.path.join(args.output_dir, 'All_result' + '.txt')
        print("----------------------final_result_file------------------------")
        print(final_result_file)
        with open(final_result_file, 'a') as f:
            f.write(model_result[model_name] + '\n')
            f.write('\n')


main()