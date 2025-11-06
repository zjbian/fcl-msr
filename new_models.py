# -*- coding: utf-8 -*-
# @Time    : 2020/3/30 10:57
# @Author  : Hui Wang

import torch
import torch.nn as nn
from modules import Encoder, LayerNorm
import math
import numpy as np
from torch.nn import Parameter
from torch.nn import functional as F
from torch.nn.init import xavier_uniform_, xavier_normal_

class S3RecModel(nn.Module):
    def __init__(self, args):
        super(S3RecModel, self).__init__()
        self.item_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)
        self.attribute_embeddings = nn.Embedding(args.attribute_size, args.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)
        self.item_encoder = Encoder(args)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.args = args

        # add unique dense layer for 4 losses respectively
        self.aap_norm = nn.Linear(args.hidden_size, args.hidden_size)
        self.mip_norm = nn.Linear(args.hidden_size, args.hidden_size)
        self.map_norm = nn.Linear(args.hidden_size, args.hidden_size)
        self.sp_norm = nn.Linear(args.hidden_size, args.hidden_size)
        self.criterion = nn.BCELoss(reduction='none')
        
        if hasattr(args, 'attr_task'):
            self.attr_encoder = Encoder(args)
            # self.indices = [[0]]
            # for i in range(1, self.args.item_size-1):
            #     self.indices.append(self.args.item2attr[i])
            # self.indices.append([0])
            # self.attr_table = self.get_attr_table().cuda()
        
    
        self.apply(self.init_weights)


        # multi-modal
        if hasattr(args, 'multi_modal_weight'):
            # assert args.hidden_size == args.clip_hidden_unit, f"当前多模态模式仅支持 hidden_size 为 {args.clip_hidden_unit}, 目前为{args.hidden_size}"

            self.img_embeddings = nn.Embedding(args.item_size, args.clip_hidden_unit, padding_idx=0)
            self.text_embeddings = nn.Embedding(args.item_size, args.clip_hidden_unit, padding_idx=0)
            if self.args.ablation_code != 2:
                self.img_embeddings.load_state_dict({
                    'weight': torch.tensor(args.multi_modal_weight['img_weights'], requires_grad=True).cuda().squeeze()
                })
                self.text_embeddings.load_state_dict({
                    'weight': torch.tensor(args.multi_modal_weight['text_weights'], requires_grad=True).cuda().squeeze()
                })

            # 打开梯度更新会带来提升  no
            if not args.multi_modal_update:
                self.img_embeddings.weight.requires_grad = False
                self.text_embeddings.weight.requires_grad = False
            
            # self.fusion_module = ModalityFusion()
            self.fusion_module = QKVAttention(args=args, q_dim=args.hidden_size, input_dim=512)
        
            transformer_width = args.clip_hidden_unit
            W_hidden = self.args.hidden_size
                    
            # self.W_t = nn.Parameter(torch.empty(W_hidden, transformer_width))
            # self.W_f = nn.Parameter(torch.empty(transformer_width, W_hidden))
            
            # for clip loss
            # self.W_img = nn.Parameter(torch.empty(transformer_width, W_hidden))
            # self.W_text = nn.Parameter(torch.empty(transformer_width, W_hidden))
            # nn.init.normal_(self.W_img, std=transformer_width ** -0.5)
            # nn.init.normal_(self.W_text, std=transformer_width ** -0.5)

            # self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
            self.logit_scale = nn.Parameter(torch.tensor(2.6592))
            
        if hasattr(self.args, 'MMOE'):
            #print("MMOE defined")
            # self.mmoe = mulit_gate(args)

            self.mmoe = mulit_gate(args)
            # self.mmoe = MLP_gate(args)

            # self.mmoe = mulit_gate2(args)

            # self.mmoe1 = mulit_gate(args, concat_hidden=self.args.hidden_size*4)
            # self.mmoe2 = mulit_gate(args, concat_hidden=self.args.hidden_size*4)
        if hasattr(self.args, 'MLP'):
            # print("mlp define")
            self.mlp = MLP_gate(args)
        if hasattr(self.args, 'Trans'):
            # print("mlp define")
            self.trans = Trans_gate(args)

    def clip_pretrain_loss(self, img_emb, text_emb, target_pos):

        mask = target_pos>0
        mask = mask.unsqueeze(-1).repeat(1, 1, self.args.hidden_size)
        img_emb = img_emb.masked_select(mask).view(-1, self.args.hidden_size)
        text_emb = text_emb.masked_select(mask).view(-1, self.args.hidden_size)
        # pos_mask = (target_pos>0)
        # seq_mask = pos_mask.unsqueeze(-1).repeat(1, 1, self.args.hidden_size)
        # target_pos = target_pos.masked_select(pos_mask)
        # seq_out = seq_out.masked_select(seq_mask).view(-1, self.args.hidden_size)
        # logits = torch.matmul(seq_out, test_item_emb.transpose(0, 1))

        # sequence = sequence[sequence != 0]

        # img_emb = self.img_embeddings(sequence) @ self.W_img
        # text_emb = self.img_embeddings(sequence) @ self.W_text

        img_emb = img_emb / img_emb.norm(p=2, dim = -1, keepdim = True)
        text_emb = text_emb / text_emb.norm(p=2 , dim = -1, keepdim = True)

        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * img_emb @ text_emb.t()
        logits_per_text = logits_per_image.t()

        # label = torch.arange(sequence.shape[0]).cuda()
        label = torch.arange(img_emb.shape[0],dtype=torch.long).cuda()
        # print(label.shape, logits_per_image.shape)
        img_loss = torch.nn.CrossEntropyLoss()(logits_per_image, label)
        text_loss = torch.nn.CrossEntropyLoss()(logits_per_text, label)

        return (img_loss + text_loss) / 2
        return text_loss

    # AAP
    def associated_attribute_prediction(self, sequence_output, attribute_embedding):
        '''
        :param sequence_output: [B L H]
        :param attribute_embedding: [arribute_num H]
        :return: scores [B*L tag_num]
        '''
        sequence_output = self.aap_norm(sequence_output) # [B L H]
        sequence_output = sequence_output.view([-1, self.args.hidden_size, 1]) # [B*L H 1]
        # [tag_num H] [B*L H 1] -> [B*L tag_num 1]
        score = torch.matmul(attribute_embedding, sequence_output)
        return torch.sigmoid(score.squeeze(-1)) # [B*L tag_num]

    # MIP sample neg items
    def masked_item_prediction(self, sequence_output, target_item):
        '''
        :param sequence_output: [B L H]
        :param target_item: [B L H]
        :return: scores [B*L]
        '''
        sequence_output = self.mip_norm(sequence_output.view([-1,self.args.hidden_size])) # [B*L H]
        target_item = target_item.view([-1,self.args.hidden_size]) # [B*L H]
        score = torch.mul(sequence_output, target_item) # [B*L H]
        return torch.sigmoid(torch.sum(score, -1)) # [B*L]

    # MAP
    def masked_attribute_prediction(self, sequence_output, attribute_embedding):
        sequence_output = self.map_norm(sequence_output)  # [B L H]
        sequence_output = sequence_output.view([-1, self.args.hidden_size, 1])  # [B*L H 1]
        # [tag_num H] [B*L H 1] -> [B*L tag_num 1]
        score = torch.matmul(attribute_embedding, sequence_output)
        return torch.sigmoid(score.squeeze(-1)) # [B*L tag_num]

    # SP sample neg segment
    def segment_prediction(self, context, segment):
        '''
        :param context: [B H]
        :param segment: [B H]
        :return:
        '''
        context = self.sp_norm(context)
        score = torch.mul(context, segment) # [B H]
        return torch.sigmoid(torch.sum(score, dim=-1)) # [B]

    #
    def add_position_embedding(self, sequence):

        seq_length = sequence.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device)
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)
        item_embeddings = self.item_embeddings(sequence)
        position_embeddings = self.position_embeddings(position_ids)

        sequence_emb = item_embeddings + position_embeddings

        # if self.args.debug_code == 0:
        #     sequence_emb += img_emb + text_emb

        sequence_emb = self.LayerNorm(sequence_emb)
        sequence_emb = self.dropout(sequence_emb)

        # 在原始的emb经过 LN 和 dropout 后在添加img和text emb效果更好。
        # if self.args.debug_code == 1:
        # if hasattr(self.args, 'multi_modal_weight'):
        #     img_emb = self.img_embeddings(sequence)
        #     text_emb = self.text_embeddings(sequence)
        #     # sequence_emb += self.LayerNorm(self.args.lambda1*img_emb @ self.W_f + self.args.lambda2*text_emb @ self.W_f)
        #     sequence_emb += self.args.lambda1*img_emb @ self.W_f + self.args.lambda2*text_emb @ self.W_f
        # sequence_emb = self.LayerNorm(sequence_emb)
        # sequence_emb = self.dropout(sequence_emb)


        return sequence_emb

    def pretrain(self, attributes, masked_item_sequence, pos_items,  neg_items,
                  masked_segment_sequence, pos_segment, neg_segment):

        # Encode masked sequence
        sequence_emb = self.add_position_embedding(masked_item_sequence)
        sequence_mask = (masked_item_sequence == 0).float() * -1e8
        sequence_mask = torch.unsqueeze(torch.unsqueeze(sequence_mask, 1), 1)

        encoded_layers = self.item_encoder(sequence_emb,
                                          sequence_mask,
                                          output_all_encoded_layers=True)
        # [B L H]
        sequence_output = encoded_layers[-1]

        attribute_embeddings = self.attribute_embeddings.weight
        # AAP
        aap_score = self.associated_attribute_prediction(sequence_output, attribute_embeddings)
        aap_loss = self.criterion(aap_score, attributes.view(-1, self.args.attribute_size).float())
        # only compute loss at non-masked position
        aap_mask = (masked_item_sequence != self.args.mask_id).float() * \
                         (masked_item_sequence != 0).float()
        aap_loss = torch.sum(aap_loss * aap_mask.flatten().unsqueeze(-1))

        # MIP
        pos_item_embs = self.item_embeddings(pos_items)
        neg_item_embs = self.item_embeddings(neg_items)
        pos_score = self.masked_item_prediction(sequence_output, pos_item_embs)
        neg_score = self.masked_item_prediction(sequence_output, neg_item_embs)
        mip_distance = torch.sigmoid(pos_score - neg_score)
        mip_loss = self.criterion(mip_distance, torch.ones_like(mip_distance, dtype=torch.float32))
        mip_mask = (masked_item_sequence == self.args.mask_id).float()
        mip_loss = torch.sum(mip_loss * mip_mask.flatten())

        # MAP
        map_score = self.masked_attribute_prediction(sequence_output, attribute_embeddings)
        map_loss = self.criterion(map_score, attributes.view(-1, self.args.attribute_size).float())
        map_mask = (masked_item_sequence == self.args.mask_id).float()
        map_loss = torch.sum(map_loss * map_mask.flatten().unsqueeze(-1))

        # SP
        # segment context
        segment_context = self.add_position_embedding(masked_segment_sequence)
        segment_mask = (masked_segment_sequence == 0).float() * -1e8
        segment_mask = torch.unsqueeze(torch.unsqueeze(segment_mask, 1), 1)
        segment_encoded_layers = self.item_encoder(segment_context,
                                               segment_mask,
                                               output_all_encoded_layers=True)

        # take the last position hidden as the context
        segment_context = segment_encoded_layers[-1][:, -1, :]# [B H]
        # pos_segment
        pos_segment_emb = self.add_position_embedding(pos_segment)
        pos_segment_mask = (pos_segment == 0).float() * -1e8
        pos_segment_mask = torch.unsqueeze(torch.unsqueeze(pos_segment_mask, 1), 1)
        pos_segment_encoded_layers = self.item_encoder(pos_segment_emb,
                                                   pos_segment_mask,
                                                   output_all_encoded_layers=True)
        pos_segment_emb = pos_segment_encoded_layers[-1][:, -1, :]

        # neg_segment
        neg_segment_emb = self.add_position_embedding(neg_segment)
        neg_segment_mask = (neg_segment == 0).float() * -1e8
        neg_segment_mask = torch.unsqueeze(torch.unsqueeze(neg_segment_mask, 1), 1)
        neg_segment_encoded_layers = self.item_encoder(neg_segment_emb,
                                                       neg_segment_mask,
                                                       output_all_encoded_layers=True)
        neg_segment_emb = neg_segment_encoded_layers[-1][:, -1, :] # [B H]

        pos_segment_score = self.segment_prediction(segment_context, pos_segment_emb)
        neg_segment_score = self.segment_prediction(segment_context, neg_segment_emb)

        sp_distance = torch.sigmoid(pos_segment_score - neg_segment_score)

        sp_loss = torch.sum(self.criterion(sp_distance,
                                           torch.ones_like(sp_distance, dtype=torch.float32)))

        return aap_loss, mip_loss, map_loss, sp_loss

    def get_attr_table(self):
        attr_table = torch.zeros_like(self.item_embeddings.weight).cuda()
        for i, indice in enumerate(self.indices):
            attr_table[i] = self.attribute_embeddings.weight[indice].mean(0) # indice error need check run test file
        return attr_table

    # Fine tune
    # same as SASRec
    def finetune(self, input_ids, attrs):

        attention_mask = (input_ids > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2) # torch.int64
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1) # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long()

        if self.args.cuda_condition:
            subsequent_mask = subsequent_mask.cuda()

        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        sequence_emb = self.add_position_embedding(input_ids)

        if hasattr(self.args, 'multi_modal_weight'):
            img_emb = self.args.lambda1 * self.img_embeddings(input_ids)
            text_emb = self.args.lambda2 * self.text_embeddings(input_ids)
            
            # if self.args.ablation_code == 2:
            #     img_emb = torch.zeros_like(img_emb)
            #     text_emb = torch.zeros_like(text_emb)
            
            # img_and_text = self.args.lambda1*img_emb + self.args.lambda2*text_emb  # 2024.3.10 version
            img_and_text = img_emb + text_emb   # 2024.3.11 version for test 
            
            if self.args.ablation_code == 5:
                img_and_text = img_emb
            elif self.args.ablation_code == 6:
                img_and_text = text_emb


            if self.args.ablation_code != 1:                
                sequence_emb, img_and_text = self.fusion_module(sequence_emb, img_and_text)
            # sequence_emb, img_and_text = self.fusion_module(sequence_emb, img_and_text)
            # sequence_emb += img_and_text
        #     sequence_emb += self.args.lambda1*img_emb @ self.W_f + self.args.lambda2*text_emb @ self.W_f

        if hasattr(self.args, 'attr_task'):
            seq_attr = self.attribute_embeddings.weight[attrs].mean(-2)  
            seq_attr_output = self.attr_encoder(seq_attr,
                                                extended_attention_mask,
                                                output_all_encoded_layers=True)[-1] # here [-1] is important
            if self.args.ablation_code == 3:
                seq_attr_output = torch.zeros_like(seq_attr_output)

        item_encoded_layers = self.item_encoder(sequence_emb,# + seq_attr,
                                                extended_attention_mask,
                                                output_all_encoded_layers=True)

        # sequence_output = item_encoded_layers[-1]
        sequence_output = item_encoded_layers[-1] #+ img_and_text
        
        # if hasattr(self.args, 'multi_modal_weight'):
        #     img_emb = self.img_embeddings(input_ids)
        #     text_emb = self.text_embeddings(input_ids)
        #     # sequence_output += self.args.lambda1*img_emb + self.args.lambda2*text_emb
        #     sequence_output += self.LayerNorm(self.args.lambda1*img_emb @ self.W_f + self.args.lambda2*text_emb @ self.W_f)

        if hasattr(self.args, 'MMOE') and self.args.MMOE:
            #print("MMOE IS TRUE IN MODELS")
            _ ,attr_out,image_out,text_out = self.mmoe(sequence_output, seq_attr_output, [img_emb, text_emb])
            # tmp,attr_out,image_out,text_out = self.mmoe(sequence_output, seq_attr_output, [img_emb, text_emb])
            # sequence_output,attr_out,image_out,text_out = self.mmoe(sequence_output, seq_attr_output)
            # sequence_output = self.mmoe(sequence_output, seq_attr_output, img_and_text)
            # sequence_output, attr_out = self.mmoe(sequence_output, seq_attr_output)
            # sequence_output = self.mmoe(sequence_output)

            # if self.args.debug_index == 1:
            #     sequence_output,attr_out,image_out,text_out = self.mmoe1(sequence_output, attr_out, [image_out, text_out])
            #     sequence_output,attr_out,image_out,text_out = self.mmoe2(sequence_output, attr_out, [image_out, text_out])
          # ori: 1 2 3 4 5 6  # input 32,5 embed -> 32,5,64   32,5,|I|
          # train seq: 1 2 3 4 5   sequence_output: 32,5,64    5,64   1:64*64  2:64  
          # label seq: 2 3 4 5 6   loss pred 32,5 - label: 32,5 
            return sequence_output, attr_out, image_out, text_out
        elif hasattr(self.args, 'MLP') and self.args.MLP:
            print("MMOE IS false IN MODELS and MLP is TRUE")
            sequence_output ,attr_out,image_out,text_out = self.mlp(sequence_output, seq_attr_output, [img_emb, text_emb])
            return sequence_output, attr_out, image_out, text_out
        elif hasattr(self.args, 'Trans') and self.args.Trans:
            # print("sequence_output shape:", sequence_output.shape)
            # print("seq_attr_output shape:", seq_attr_output.shape)
            # print("img_emb shape:", img_emb.shape)
            # print("text_emb shape:", text_emb.shape)
            sequence_output ,attr_out,image_out,text_out = self.trans(sequence_output, seq_attr_output, [img_emb, text_emb])
            return sequence_output, attr_out, image_out, text_out
        return sequence_output

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.args.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class Expert(nn.Module):
    def __init__(self, hidden_in, hidden_out, dropout):
        super(Expert, self).__init__()
        self.l1 = nn.Linear(hidden_in, hidden_out)
        self.activation = nn.GELU()
        self.l2 = nn.Linear(hidden_out, hidden_out)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_out)
        self.apply(self.init_weights)

    def forward(self, x):
        out = self.l1(x)
        out = self.activation(out)
        out = self.l2(out)
        out = self.activation(out)
        out = self.dropout(out)
        # out = self.norm(out + x[:,:,:64])
        out = self.norm(out)
        out = nn.Softplus()(out)
        return out

    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
        
class Gate(nn.Module):
    def __init__(self, hidden, expert_num):
        super(Gate, self).__init__()
        self.l1 = nn.Linear(hidden, expert_num, bias=False)
        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, x):
        # return nn.functional.softmax(self.l1(x), -1)
        return nn.Softplus()(self.l1(x))

# TODO multi task
# TODO MLP
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = F.softmax(x, dim=-1)  # 使用 softmax 激活函数，确保输出是概率分布
        return x


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.permute(1, 0, 2)  # (seq_len, batch_size, hidden_size)
        attn_output, _ = self.attention(x, x, x)
        attn_output = self.dropout(attn_output)
        x = self.norm1(x + attn_output)
        ffn_output = self.ffn(x)
        ffn_output = self.dropout(ffn_output)
        x = self.norm2(x + ffn_output)
        x = x.permute(1, 0, 2)  # (batch_size, seq_len, hidden_size)    
        return x

class MLP_gate(nn.Module):
    def __init__(self, args, concat_hidden=None):
        super(MLP_gate, self).__init__()
        
        self.hidden = args.hidden_size
        self.dropout = args.dropout
        self.share_expert_num = args.main_expert_num  # best 4 2 2 2 
        self.attr_expert_num = args.modal_expert_num
        self.image_expert_num = args.modal_expert_num
        self.text_expert_num = args.modal_expert_num
        if concat_hidden is not None:
            self.concat_hidden = concat_hidden
        else:
            self.concat_hidden = self.hidden * 2 + 2 * 512 

        # 使用 MLP 替换 Expert 和 Gate
        self.share_mlp = MLP(self.concat_hidden, self.hidden, self.hidden, self.dropout)
        self.attr_mlp = MLP(self.concat_hidden, self.hidden, self.hidden, self.dropout)
        self.image_mlp = MLP(self.concat_hidden, self.hidden, self.hidden, self.dropout)
        self.text_mlp = MLP(self.concat_hidden, self.hidden, self.hidden, self.dropout)

    def forward(self, seq_out, seq_attr=None, img_text=None):
        img_embed, text_embed = img_text        
        x = torch.cat([seq_out, img_embed, text_embed, seq_attr], -1)
        

        # 使用 MLP 进行特征提取
        share_out = self.share_mlp(x)
        attr_out = self.attr_mlp(x)
        image_out = self.image_mlp(x)
        text_out = self.text_mlp(x)

        return share_out, attr_out, image_out, text_out


class Trans_gate(nn.Module):
    def __init__(self, args):
        super(Trans_gate, self).__init__()
        self.hidden = args.hidden_size
        self.dropout = args.dropout
        self.share_expert_num = args.main_expert_num
        self.attr_expert_num = args.modal_expert_num
        self.image_expert_num = args.modal_expert_num
        self.text_expert_num = args.modal_expert_num
        self.concat_hidden = self.hidden * 2 + 2 * 512  # 计算拼接后的隐藏层维度
        # 使用 TransformerBlock 替换原来的 Expert 模块
        self.share_transformers = TransformerBlock(self.concat_hidden, args.num_heads, self.dropout)

        # self.main_gate = Gate(self.concat_hidden, self.share_expert_num)

        self.attr_transformers = TransformerBlock(self.concat_hidden, args.num_heads, self.dropout) 

        # self.attr_gate = Gate(self.concat_hidden, self.share_expert_num + self.attr_expert_num)

        self.image_transformers = TransformerBlock(self.concat_hidden, args.num_heads, self.dropout) 

        # self.image_gate = Gate(self.concat_hidden, self.share_expert_num + self.image_expert_num)

        self.text_transformers = TransformerBlock(self.concat_hidden, args.num_heads, self.dropout) 
        # self.text_gate = Gate(self.concat_hidden, self.share_expert_num + self.text_expert_num)

        # 新增线性层用于将不同模态数据的维度统一
        # self.sequence_expander = nn.Linear(64, self.concat_hidden)
        # self.seq_attr_expander = nn.Linear(64, self.concat_hidden)
        # self.image_emb_reducer = nn.Linear(512, self.concat_hidden)
        # self.text_emb_reducer = nn.Linear(512, self.concat_hidden)

        # 最终输出的线性层，将结果降维到所需的输出维度
        self.share_out_reducer = nn.Linear(self.concat_hidden, 64)
        self.attr_out_reducer = nn.Linear(self.concat_hidden, 64)
        self.image_out_reducer = nn.Linear(self.concat_hidden, 64)
        self.text_out_reducer = nn.Linear(self.concat_hidden, 64)

    def forward(self, sequence_output, seq_attr_output, img_text):
        img_emb, text_emb = img_text
        # 将不同模态的数据维度统一
        # sequence_output = self.sequence_expander(sequence_output)
        # seq_attr_output = self.seq_attr_expander(seq_attr_output)
        # img_emb = self.image_emb_reducer(img_emb)
        # text_emb = self.text_emb_reducer(text_emb)

        x = torch.cat([sequence_output, seq_attr_output, img_emb, text_emb], dim=-1)

        share_out = self.share_transformers(x)
        attr_out = self.attr_transformers(x)
        image_out = self.image_transformers(x)
        text_out = self.text_transformers(x)

 

        # 降维操作，将结果转换为所需的输出维度
        share_out = self.share_out_reducer(share_out)
        attr_out = self.attr_out_reducer(attr_out)
        image_out = self.image_out_reducer(image_out)
        text_out = self.text_out_reducer(text_out)

        return share_out, attr_out, image_out, text_out



class mulit_gate(nn.Module):
    def __init__(self, args, concat_hidden=None):
        super(mulit_gate, self).__init__()
        
        self.hidden = args.hidden_size
        self.dropout = args.dropout
        self.share_expert_num = args.main_expert_num  # best 4 2 2 2 
        self.attr_expert_num = args.modal_expert_num
        self.image_expert_num = args.modal_expert_num
        self.text_expert_num = args.modal_expert_num
        if concat_hidden is not None:
            self.concat_hidden = concat_hidden
        else:
            self.concat_hidden = self.hidden*2 + 2*512 

        self.share_experts = nn.ModuleList([
            Expert(self.concat_hidden, self.hidden, self.dropout) for i in range(self.share_expert_num)
        ])
        self.main_gate = Gate(self.concat_hidden, self.share_expert_num)
 
        self.attr_experts = nn.ModuleList([
            Expert(self.concat_hidden, self.hidden, self.dropout) for i in range(self.attr_expert_num)
        ])
        self.attr_gate = Gate(self.concat_hidden, self.share_expert_num + self.attr_expert_num)

        self.image_experts = nn.ModuleList([
            Expert(self.concat_hidden, self.hidden, self.dropout) for i in range(self.image_expert_num)
        ])
        self.image_gate = Gate(self.concat_hidden, self.share_expert_num + self.image_expert_num)

        self.text_experts = nn.ModuleList([
            Expert(self.concat_hidden, self.hidden, self.dropout) for i in range(self.text_expert_num)
        ])
        self.text_gate = Gate(self.concat_hidden, self.share_expert_num + self.text_expert_num)

    def forward(self, seq_out, seq_attr=None, img_text=None):
        img_embed, text_embed = img_text        
        x = torch.cat([seq_out, img_embed, text_embed, seq_attr], -1)

        # x = torch.cat([seq_out, seq_attr], -1)
                
        # x = seq_out
        share_res = []
        for expert in self.share_experts:
            share_res.append(expert(x))
        share_res = torch.stack(share_res, -2) # N,L,expert_num,D
        share_gate = self.main_gate(x) # N,L,expert_num
        share_out = torch.einsum('nec,necd ->necd', share_gate, share_res).sum(-2)        

        attr_res = []
        for expert in self.attr_experts:
            attr_res.append(expert(x))
        attr_res = torch.stack(attr_res, -2)
        attr_res = torch.cat([attr_res, share_res], -2)
        attr_gate = self.attr_gate(x)
        attr_out = torch.einsum('nec,necd ->necd', attr_gate, attr_res).sum(-2)        

        image_res = []
        for expert in self.image_experts:
            image_res.append(expert(x))
        image_res = torch.stack(image_res, -2)
        image_res = torch.cat([image_res, share_res], -2)
        image_gate = self.image_gate(x)
        image_out = torch.einsum('nec,necd ->necd', image_gate, image_res).sum(-2)

        text_res = []
        for expert in self.text_experts:
            text_res.append(expert(x))
        text_res = torch.stack(text_res, -2)
        text_res = torch.cat([text_res, share_res], -2)
        text_gate = self.text_gate(x)
        text_out = torch.einsum('nec,necd ->necd', text_gate, text_res).sum(-2)                                                                            

        return share_out, attr_out, image_out, text_out


class ModalityFusion(nn.Module):
    def __init__(self, input_dim1=512, input_dim2=64, output_dim=64):
        super(ModalityFusion, self).__init__()
        self.fc1 = nn.Linear(input_dim1, 256)
        self.fc2 = nn.Linear(input_dim2, 256)
        self.fc3 = nn.Linear(256+64, output_dim)

    def forward(self, x1, x2):
        x1 = torch.relu(self.fc1(x1))
        # x2 = torch.relu(self.fc2(x2))
        x = torch.cat((x1, x2), dim=2)
        x = torch.relu(self.fc3(x))
        return x

class QKVAttention(nn.Module):
    def __init__(self, args, q_dim=64, input_dim=512, num_heads=8):
        super(QKVAttention, self).__init__()
        self.args = args
        self.input_dim = input_dim
        self.num_heads = num_heads

        # Linear transformations for Q, K, and V
        self.linear_q = nn.Linear(q_dim, q_dim)
        self.linear_k = nn.Linear(input_dim, q_dim)
        self.linear_v = nn.Linear(input_dim, q_dim)

        # Final linear transformation for the output
        self.linear_out = nn.Linear(q_dim, q_dim)

    def forward(self, x, v):
        batch_size, seq_len, input_dim = x.size()

        # Linear transformations for Q, K, and V
        q = self.linear_q(x).view(batch_size, seq_len, self.num_heads, input_dim // self.num_heads).transpose(1, 2)
        k = self.linear_k(v).view(batch_size, seq_len, self.num_heads, input_dim // self.num_heads).transpose(1, 2)
        v = self.linear_v(v).view(batch_size, seq_len, self.num_heads, input_dim // self.num_heads).transpose(1, 2)

        # Scaled Dot-Product Attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (input_dim // self.num_heads) ** 0.5
        attn_probs = torch.nn.functional.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_probs, v)

        # Concatenate and linear transformation for the output
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, input_dim)
        output = self.linear_out(attn_output)

        if self.args.ablation_code == 1:
            return x, output
        return output + x, output



class mulit_gate2(nn.Module):
    def __init__(self, args, concat_hidden=None):
        super(mulit_gate2, self).__init__()
        
        self.hidden = args.hidden_size
        self.dropout = args.dropout
        self.share_expert_num = 8  # best 4 2 2 2 
        self.attr_expert_num = 4
        self.image_expert_num = 4
        self.text_expert_num = 4
        if concat_hidden is not None:
            self.concat_hidden = concat_hidden
        else:
            self.concat_hidden = self.hidden*2 + 2*512 

        self.share_experts = nn.ModuleList([
            Expert(self.concat_hidden, self.hidden, self.dropout) for i in range(self.share_expert_num)
        ])
        
        self.main_gate = Gate(self.concat_hidden, self.share_expert_num +
                                        self.attr_expert_num + self.image_expert_num + self.text_expert_num)
    
        self.attr_experts = nn.ModuleList([
            Expert(self.concat_hidden, self.hidden, self.dropout) for i in range(self.attr_expert_num)
        ])
        self.attr_gate = Gate(self.concat_hidden, self.attr_expert_num)

        self.image_experts = nn.ModuleList([
            Expert(self.concat_hidden, self.hidden, self.dropout) for i in range(self.image_expert_num)
        ])
        self.image_gate = Gate(self.concat_hidden, self.image_expert_num)

        self.text_experts = nn.ModuleList([
            Expert(self.concat_hidden, self.hidden, self.dropout) for i in range(self.text_expert_num)
        ])
        self.text_gate = Gate(self.concat_hidden, self.text_expert_num)

    def forward(self, seq_out, seq_attr=None, img_text=None):
        img_embed, text_embed = img_text        
        x = torch.cat([seq_out, img_embed, text_embed, seq_attr], -1)

        # x = torch.cat([seq_out, seq_attr], -1)
                
        # x = seq_out
        attr_res = []
        for expert in self.attr_experts:
            attr_res.append(expert(x))
        attr_res = torch.stack(attr_res, -2)
        # attr_res = torch.cat([attr_res, share_res], -2)
        attr_gate = self.attr_gate(x)
        attr_out = torch.einsum('nec,necd ->necd', attr_gate, attr_res).sum(-2)        

        image_res = []
        for expert in self.image_experts:
            image_res.append(expert(x))
        image_res = torch.stack(image_res, -2)
        # image_res = torch.cat([image_res, share_res], -2)
        image_gate = self.image_gate(x)
        image_out = torch.einsum('nec,necd ->necd', image_gate, image_res).sum(-2)

        text_res = []
        for expert in self.text_experts:
            text_res.append(expert(x))
        text_res = torch.stack(text_res, -2)
        # text_res = torch.cat([text_res, share_res], -2)
        text_gate = self.text_gate(x)
        text_out = torch.einsum('nec,necd ->necd', text_gate, text_res).sum(-2)    

        share_res = []
        for expert in self.share_experts:
            share_res.append(expert(x))
        share_res = torch.stack(share_res, -2) # N,L,expert_num,D
        share_res = torch.cat([share_res, attr_res, image_res, text_res], -2) # N,L,expert_num,D+D])
        share_gate = self.main_gate(x) # N,L,expert_num
        share_out = torch.einsum('nec,necd ->necd', share_gate, share_res).sum(-2)                                                                               

        return share_out, attr_out, image_out, text_out




