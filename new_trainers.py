import numpy as np
import tqdm
import random

import torch
import torch.nn as nn
from torch.optim import Adam

from utils import recall_at_k, ndcg_k, get_metric


class Trainer:
    def __init__(self, model, train_dataloader,
                 eval_dataloader,
                 test_dataloader, args):

        self.args = args
        self.cuda_condition = torch.cuda.is_available() and not self.args.no_cuda
        self.device = torch.device("cuda" if self.cuda_condition else "cpu")

        self.model = model
        if self.cuda_condition:
            self.model.cuda()
        self.temperature = self.args.temperature
        # Setting the train and test data loader
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.test_dataloader = test_dataloader

        # self.data_name = self.args.data_name
        betas = (self.args.adam_beta1, self.args.adam_beta2)
        self.optim = Adam(self.model.parameters(), lr=self.args.lr, betas=betas, weight_decay=self.args.weight_decay)

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))
        self.criterion = nn.BCELoss()

    def train(self, epoch, verbose = False):
        return self.iteration(epoch, self.train_dataloader, verbose=verbose)

    def valid(self, epoch, full_sort=False, verbose = False):
        return self.iteration(epoch, self.eval_dataloader, full_sort, train=False, verbose = verbose)

    def test(self, epoch, full_sort=False, verbose = False):

        # if hasattr(self.args, 'gap'):
        #     if self.args.gap != 0:
        #         user_seq = self.test_dataloader.dataset.user_seq
        #         from utils import generate_rating_matrix_test
        #         self.args.train_matrix = generate_rating_matrix_test(user_seq, len(user_seq), self.args.item_size)

        return self.iteration(epoch, self.test_dataloader, full_sort, train=False, verbose = verbose)

    def iteration(self, epoch, dataloader, full_sort=False, train=True):
        raise NotImplementedError

    def get_sample_scores(self, epoch, pred_list, verbose = False):
        pred_list = (-pred_list).argsort().argsort()[:, 0]
        HIT_1, NDCG_1, MRR = get_metric(pred_list, 1)
        HIT_5, NDCG_5, MRR = get_metric(pred_list, 5)
        HIT_15, NDCG_15, MRR = get_metric(pred_list, 15)
        HIT_10, NDCG_10, MRR = get_metric(pred_list, 10)
        post_fix = {
            "Epoch": epoch,
            "HIT@1": '{:.4f}'.format(HIT_1), "NDCG@1": '{:.4f}'.format(NDCG_1),
            "HIT@5": '{:.4f}'.format(HIT_5), "NDCG@5": '{:.4f}'.format(NDCG_5),
            "HIT@10": '{:.4f}'.format(HIT_10), "NDCG@10": '{:.4f}'.format(NDCG_10),
            "HIT@15": '{:.4f}'.format(HIT_15), "NDCG@15": '{:.4f}'.format(NDCG_15),
            "MRR": '{:.4f}'.format(MRR),
        }
        if verbose:
            print(post_fix)
        with open(self.args.log_file, 'a') as f:
            f.write(str(post_fix) + '\n')
        return [HIT_1, NDCG_1, HIT_5, NDCG_5, HIT_10, NDCG_10, HIT_15, NDCG_15,MRR], str(post_fix)

    def get_full_sort_score(self, epoch, answers, pred_list, verbose = False):
        recall, ndcg = [], []
        for k in [5,10,15,50]:
            recall.append(recall_at_k(answers, pred_list, k))
            ndcg.append(ndcg_k(answers, pred_list, k))
        post_fix = {
            "Epoch": epoch,
            "HIT@5": '{:.4f}'.format(recall[0]), "NDCG@5": '{:.4f}'.format(ndcg[0]),
            "HIT@10": '{:.4f}'.format(recall[1]), "NDCG@10": '{:.4f}'.format(ndcg[1]),
            "HIT@15": '{:.4f}'.format(recall[2]), "NDCG@15": '{:.4f}'.format(ndcg[2]),
            "HIT@50": '{:.4f}'.format(recall[3]), "NDCG@50": '{:.4f}'.format(ndcg[3])
        }
        if verbose:
            print(post_fix)
            if self.args.writer:
                self.args.writer.add_scalar('HIT@5',recall[0], epoch)
                self.args.writer.add_scalar('HIT@10',recall[1], epoch)
                self.args.writer.add_scalar('HIT@15',recall[2], epoch)
                self.args.writer.add_scalar('NDCG@5',ndcg[0], epoch)
                self.args.writer.add_scalar('NDCG@10',ndcg[1], epoch)
                self.args.writer.add_scalar('NDCG@15',recall[2], epoch)

        with open(self.args.log_file, 'a') as f:
            f.write(str(post_fix) + '\n')        
        return [recall[0], ndcg[0], recall[1], ndcg[1], recall[2],ndcg[2],recall[3],ndcg[3]], str(post_fix)

    def save(self, file_name):
        torch.save(self.model.cpu().state_dict(), file_name)
        self.model.to(self.device)

    def load(self, file_name):
        self.model.load_state_dict(torch.load(file_name), strict = False)

    def bce_loss(self, seq_out, pos_ids, neg_ids, sample_num = 0):
        # [batch seq_len hidden_size]
        batch, seq_len, hidden_size = seq_out.shape
        if sample_num != 0:
            sample_ids = np.random.choice(self.args.item_size, (batch, seq_len, sample_num))
            tmp_targets = np.repeat(np.array(pos_ids.tolist()), sample_num).reshape(batch, seq_len, sample_num)
            mask = tmp_targets == sample_ids
            sample_ids[mask] = self.args.mask_id
            neg_ids = torch.from_numpy(sample_ids).to(pos_ids.device)

            pos_emb = self.model.item_embeddings(pos_ids)
            neg_emb = self.model.item_embeddings(neg_ids)
            # [batch*seq_len hidden_size]
            pos = pos_emb.view(-1, pos_emb.shape[-1])
            # [batch*seq_len, sample_num, hidden_size]
            neg = neg_emb.view(-1, sample_num, neg_emb.shape[-1])
            seq_emb = seq_out.view(-1, self.args.hidden_size) # [batch*seq_len hidden_size]
            seq_emb_neg = seq_out.unsqueeze(2).repeat(1,1,sample_num,1).view(-1, sample_num, self.args.hidden_size)
            pos_logits = torch.sum(pos * seq_emb, -1) # [batch*seq_len]
            neg_logits = torch.sum(neg * seq_emb_neg, -1) #[batch*seq_len, sample_num]
            istarget = (pos_ids > 0).view(pos_ids.size(0) * self.model.args.max_seq_length).float() # [batch*seq_len]
            istarget_neg = istarget.unsqueeze(-1).repeat(1, sample_num)
            #TODO change this similarity
            loss = torch.sum(
                - torch.log(torch.sigmoid(pos_logits) + 1e-24) * istarget -
                (torch.log(1 - torch.sigmoid(neg_logits) + 1e-24) * istarget_neg).mean()
            ) / torch.sum(istarget)
            return loss
        else:
            pos_emb = self.model.item_embeddings(pos_ids)
            neg_emb = self.model.item_embeddings(neg_ids)
            # [batch*seq_len hidden_size]
            pos = pos_emb.view(-1, pos_emb.shape[-1])
            # [batch*seq_len*sample_num hidden_size]
            neg = neg_emb.view(-1, neg_emb.shape[-1])
            seq_emb = seq_out.view(-1, self.args.hidden_size) # [batch*seq_len hidden_size]
            pos_logits = torch.sum(pos * seq_emb, -1) # [batch*seq_len]
            neg_logits = torch.sum(neg * seq_emb, -1)
            istarget = (pos_ids > 0).view(pos_ids.size(0) * self.model.args.max_seq_length).float() # [batch*seq_len]
            #TODO change this similarity
            loss = torch.sum(
                - torch.log(torch.sigmoid(pos_logits) + 1e-24) * istarget -
                torch.log(1 - torch.sigmoid(neg_logits) + 1e-24) * istarget
            ) / torch.sum(istarget)
            return loss

    def cce_loss(self, seq_out, target_pos):  #seq_out: 32, 5, 64

        
        test_item_emb = self.model.item_embeddings.weight # |I|,64  # 1~10 1: 64

        # if hasattr(self.args, 'multi_modal_weight'):

        #     if self.args.debug_code == 0:
        #         tmp_emb = self.model.img_embeddings.weight + self.model.text_embeddings.weight
        #     elif self.args.debug_code == 1:
        #         tmp_emb = self.args.lambda1*self.model.img_embeddings.weight \
        #                 + self.args.lambda2*self.model.text_embeddings.weight
            
        #     tmp_emb += test_item_emb
        #     test_item_emb = tmp_emb

        pos_mask = (target_pos>0)
        seq_mask = pos_mask.unsqueeze(-1).repeat(1, 1, self.args.hidden_size)
        target_pos = target_pos.masked_select(pos_mask)

        seq_out = seq_out.masked_select(seq_mask).view(-1, self.args.hidden_size)
        
        if self.temperature > 0:
                seq_out = nn.functional.normalize(seq_out, dim=-1)
                test_item_emb = nn.functional.normalize(test_item_emb, dim=-1)
            
        logits = torch.matmul(seq_out, test_item_emb.transpose(0, 1))  #32,5,|I|
          #target_pos 32,5
        if self.temperature > 0:
            logits /= self.temperature
        
        # logits = torch.matmul(seq_out, test_item_emb.transpose(0, 1))
        loss = nn.CrossEntropyLoss()(logits, target_pos)
        return  loss

    def attr_loss(self, attr_out, target_attr):
        test_item_emb = self.model.attribute_embeddings.weight

        pos_mask = (target_attr > 0)
        seq_mask = pos_mask.unsqueeze(-1).repeat(1, 1, 1, self.args.hidden_size)
        attr_out = attr_out.unsqueeze(-2).repeat(1, 1, 10, 1)
        attr_out = attr_out.masked_select(seq_mask).view(-1, self.args.hidden_size)
        logits = torch.matmul(attr_out, test_item_emb.transpose(0, 1))
        # logits = torch.sigmoid(logits)
        target_attr = target_attr.masked_select(pos_mask)
        # target_attr = nn.functional.one_hot(target_attr, num_classes=self.args.attribute_size).float()
        # loss = nn.BCELoss()(logits, target_attr)
        loss = nn.CrossEntropyLoss()(logits, target_attr)
        return loss


    def bpr_loss(self, seq_out, target_pos, sample_num, use_softmax=False):
        rating_pred = self.predict_full(seq_out)
        batch_size = seq_out.shape[0]
        targets = target_pos[:,-1]   
        pos = self.model.item_embeddings(targets)
        pos_logits = torch.sum(pos*seq_out, -1).unsqueeze(-1)

        sample_ids = np.random.choice(self.args.item_size, (batch_size,sample_num)) #, replace = False)
        tmp_targets = np.repeat(np.array(targets.tolist()), sample_num).reshape(batch_size,sample_num)
        mask = tmp_targets == sample_ids
        sample_ids[mask] = self.args.mask_id
        sample_ids = torch.from_numpy(sample_ids).to(targets.device)
        neg = self.model.item_embeddings(sample_ids)
        neg_logits = torch.sum(seq_out.unsqueeze(-1)*neg.transpose(2,1), -1)

        if use_softmax: # BPR-max
            coeff = torch.softmax(neg_logits, -1)
            loss = -torch.log(1e-10 + coeff*torch.sigmoid(pos_logits-neg_logits)).mean()
        else:
            loss = -torch.log(1e-10 + torch.sigmoid(pos_logits-neg_logits)).mean()
        return loss

    def top1_loss(self, seq_out, target_pos, sample_num, use_softmax=False):
        rating_pred = self.predict_full(seq_out)
        batch_size = seq_out.shape[0]
        targets = target_pos[:,-1]   
        pos = self.model.item_embeddings(targets)
        pos_logits = torch.sum(pos*seq_out, -1).unsqueeze(-1)

        sample_ids = np.random.choice(self.args.item_size, (batch_size,sample_num)) #, replace = False)
        tmp_targets = np.repeat(np.array(targets.tolist()), sample_num).reshape(batch_size,sample_num)
        mask = tmp_targets == sample_ids
        sample_ids[mask] = self.args.mask_id
        sample_ids = torch.from_numpy(sample_ids).to(targets.device)
        neg = self.model.item_embeddings(sample_ids)
        neg_logits = torch.sum(seq_out.unsqueeze(-1)*neg.transpose(2,1), -1)

        if use_softmax: # TOP1-max
            coeff = torch.softmax(neg_logits, -1)
            loss = (coeff*(torch.sigmoid(neg_logits-pos_logits) + torch.pow(neg_logits, 2))).mean()
        else:
            loss = torch.sigmoid(neg_logits-pos_logits).mean() + torch.pow(neg_logits, 2).mean()
        return loss

    def ce_loss(self, seq_out, target_pos):
        seq_out = seq_out[:,-1,:]
        target_pos = target_pos[:,-1]
        test_item_emb = self.model.item_embeddings.weight
        # logits = torch.matmul(seq_out, test_item_emb.transpose(0, 1))
        
        if self.temperature > 0:
            seq_out = nn.functional.normalize(seq_out, dim=-1)
            test_item_emb = nn.functional.normalize(test_item_emb, dim=-1)
            
        logits = torch.matmul(seq_out, test_item_emb.transpose(0, 1))
        
        if self.temperature > 0:
            logits /= self.temperature
        
        loss = nn.CrossEntropyLoss()(logits, target_pos)
        return  loss

    def mlm_loss(self, seq_out, target_pos):
        test_item_emb = self.model.item_embeddings.weight
        pos_mask = (target_pos>0)
        seq_mask = pos_mask.unsqueeze(-1).repeat(1, 1, self.args.hidden_size)
        target_pos = target_pos.masked_select(pos_mask)
        seq_out = seq_out.masked_select(seq_mask).view(-1, self.args.hidden_size)
        logits = torch.matmul(seq_out, test_item_emb.transpose(0, 1))
        loss = nn.CrossEntropyLoss()(logits, target_pos)
        return  loss

    def predict_sample(self, seq_out, test_neg_sample):
        # [batch 100 hidden_size]
        test_item_emb = self.model.item_embeddings(test_neg_sample)
        # [batch hidden_size]
        test_logits = torch.bmm(test_item_emb, seq_out.unsqueeze(-1)).squeeze(-1)  # [B 100]
        return test_logits

    def predict_full(self, seq_out):
        # [item_num hidden_size]
        test_item_emb = self.model.item_embeddings.weight
        # [batch hidden_size ]
        rating_pred = torch.matmul(seq_out, test_item_emb.transpose(0, 1))
        return rating_pred

class PretrainTrainer(Trainer):

    def __init__(self, model,
                 train_dataloader,
                 eval_dataloader,
                 test_dataloader, args):
        
        super(PretrainTrainer, self).__init__(
            model,
            train_dataloader,
            eval_dataloader,
            test_dataloader, args
        )

    def pretrain(self, epoch, pretrain_dataloader):

        desc = f'AAP-{self.args.aap_weight}-' \
               f'MIP-{self.args.mip_weight}-' \
               f'MAP-{self.args.map_weight}-' \
               f'SP-{self.args.sp_weight}'

        pretrain_data_iter = tqdm.tqdm(enumerate(pretrain_dataloader),
                                       desc=f"{self.args.model_name}-{self.args.data_name} Epoch:{epoch}",
                                       total=len(pretrain_dataloader),
                                       bar_format="{l_bar}{r_bar}")

        self.model.train()
        aap_loss_avg = 0.0
        mip_loss_avg = 0.0
        map_loss_avg = 0.0
        sp_loss_avg = 0.0

        for i, batch in pretrain_data_iter:
            # 0. batch_data will be sent into the device(GPU or CPU)
            batch = tuple(t.to(self.device) for t in batch)
            attributes, masked_item_sequence, pos_items, neg_items, \
            masked_segment_sequence, pos_segment, neg_segment = batch

            aap_loss, mip_loss, map_loss, sp_loss = self.model.pretrain(attributes,
                                            masked_item_sequence, pos_items, neg_items,
                                            masked_segment_sequence, pos_segment, neg_segment)

            joint_loss = self.args.aap_weight * aap_loss + \
                         self.args.mip_weight * mip_loss + \
                         self.args.map_weight * map_loss + \
                         self.args.sp_weight * sp_loss

            self.optim.zero_grad()
            joint_loss.backward()
            self.optim.step()

            aap_loss_avg += aap_loss.item()
            mip_loss_avg += mip_loss.item()
            map_loss_avg += map_loss.item()
            sp_loss_avg += sp_loss.item()

        num = len(pretrain_data_iter) * self.args.pre_batch_size
        post_fix = {
            "epoch": epoch,
            "aap_loss_avg": '{:.4f}'.format(aap_loss_avg /num),
            "mip_loss_avg": '{:.4f}'.format(mip_loss_avg /num),
            "map_loss_avg": '{:.4f}'.format(map_loss_avg / num),
            "sp_loss_avg": '{:.4f}'.format(sp_loss_avg / num),
        }
        print(desc)
        print(str(post_fix))
        with open(self.args.log_file, 'a') as f:
            f.write(str(desc) + '\n')
            f.write(str(post_fix) + '\n')

class FinetuneTrainer(Trainer):

    def __init__(self, model,
                 train_dataloader,
                 eval_dataloader,
                 test_dataloader, args):
        super(FinetuneTrainer, self).__init__(
            model,
            train_dataloader,
            eval_dataloader,
            test_dataloader, args
        )

    def iteration(self, epoch, dataloader, full_sort=False, train=True, verbose = False):

        str_code = "train" if train else "test"

        # Setting the tqdm progress bar

        rec_data_iter = tqdm.tqdm(enumerate(dataloader),
                                  desc="Recommendation EP_%s:%d" % (str_code, epoch),
                                  total=len(dataloader),
                                  bar_format="{l_bar}{r_bar}")
        if train:
            self.model.train()
            rec_avg_loss = 0.0
            rec_cur_loss = 0.0
            avg_loss1 = 0.0
            avg_loss2 = 0.0
            avg_loss3 = 0.0
            cur_loss1 = 0.0
            cur_loss2 = 0.0
            cur_loss3 = 0.0


            for i, batch in rec_data_iter:
                # 0. batch_data will be sent into the device(GPU or CPU)
                batch = tuple(t.to(self.device) for t in batch)
                _, input_ids, target_pos, target_neg, _, attrs, target_attr = batch
                # Binary cross_entropy

                if self.args.loss_type == 'MLM':
                    mask = torch.rand(input_ids.size(0), input_ids.size(1)) < self.args.mask_prod
                    mask = mask.to(target_pos.device)
                    target_pos = target_pos.masked_fill(~mask, 0)
                    input_ids = input_ids.masked_fill(mask, 0)      

                finetune_called = False  # 标志，确保只调用一次 finetune
                if self.args.Ours:
                    if hasattr(self.args, 'MMOE') and self.args.MMOE and not finetune_called:
                        # print("hasattr--->MMOE",self.args.MMOE)
                        sequence_output, attr_out, image_out, text_out = self.model.finetune(input_ids, attrs)
                        finetune_called = True
                    elif hasattr(self.args, 'MLP') and self.args.MLP and not finetune_called:
                        # print("hasattr--->MLP",self.args.MLP)
                        print("hasattr--->MLP")
                        sequence_output, attr_out, image_out, text_out = self.model.finetune(input_ids, attrs)
                        # print(sequence_output.shape)
                        finetune_called = True
                    elif hasattr(self.args, 'Trans') and self.args.Trans and not finetune_called:
                        print("hasattr--->Transformer")
                        # print(input_ids.shape)
                        sequence_output, attr_out, image_out, text_out = self.model.finetune(input_ids, attrs)
                        # print(sequence_output.shape)
                        finetune_called = True
                    elif not finetune_called:
                        sequence_output = self.model.finetune(input_ids, attrs)
                else:
                    sequence_output = self.model.finetune(input_ids, attrs)              


                # if self.args.Ours:
                #     if hasattr(self.args, 'MMOE'):
                #         # print("hasattr--->MMOE")
                #         if self.args.MMOE:
                #             sequence_output, attr_out, image_out, text_out = self.model.finetune(input_ids, attrs)
                #     if hasattr(self.args, 'MLP'):
                #         # print("hasattr--->MLP")
                #         if self.args.MLP:
                #             sequence_output, attr_out, image_out, text_out = self.model.finetune(input_ids, attrs)
                #     else:
                #         sequence_output = self.model.finetune(input_ids, attrs)
                # else:
                #     sequence_output = self.model.finetune(input_ids, attrs)

                # if self.args.loss_type == 'BCE':
                #     loss = self.bce_loss(sequence_output, target_pos, target_neg, self.args.sample_num)
                # if self.args.loss_type == 'CCE':  # Ours
                ## CCEloss---nip
                loss1 = self.cce_loss(sequence_output, target_pos)
                # elif self.args.loss_type == 'BPR':
                #     loss = self.bpr_loss(sequence_output[:, -1, :], 
                #                         target_pos, 
                #                         sample_num = self.args.sample_num)
                # elif self.args.loss_type == 'BPR-max':
                #     loss = self.bpr_loss(sequence_output[:, -1, :], 
                #                         target_pos, 
                #                         sample_num = self.args.sample_num,
                #                         use_softmax = True)
                # elif self.args.loss_type == 'TOP1':
                #     loss = self.top1_loss(sequence_output[:, -1, :], 
                #                         target_pos, 
                #                         sample_num = self.args.sample_num)
                # elif self.args.loss_type == 'TOP1-max':
                #     loss = self.top1_loss(sequence_output[:, -1, :], 
                #                         target_pos, 
                #                         sample_num = self.args.sample_num,
                #                         use_softmax = True)
                # elif self.args.loss_type == 'CE':
                #     loss = self.ce_loss(sequence_output, target_pos)
                # elif self.args.loss_type == 'MLM':
                #     loss = self.mlm_loss(sequence_output, target_pos)
                # elif  self.args.loss_type == 'attr_loss':
                #     loss = self.attr_loss(attr_out, target_attr)
                #TODO add Lambdarankloss 
                
                # if hasattr(self.args, 'multi_modal_weight') and hasattr(self.args, 'MMOE'):
                #     if self.args.attr_task == True and self.args.MMOE == True:
                #     # if i % 10 == 0 :
                #         loss3 = 0.1*self.model.clip_pretrain_loss(image_out, text_out, target_pos)
                loss3 = 0.1*self.model.clip_pretrain_loss(image_out, text_out, target_pos)
                # if hasattr(self.args, 'attr_task') and hasattr(self.args, 'MMOE'):
                #     if self.args.attr_task == True and self.args.MMOE == True:
                #         loss2 = self.attr_loss(attr_out, target_attr)
                loss2 = self.attr_loss(attr_out, target_attr)
                loss = loss1 + loss3 + loss2
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                rec_avg_loss += loss.item()
                rec_cur_loss = loss.item()
                
                ## loss1 for nip
                avg_loss1 += loss1.item()
                cur_loss1 = loss1.item()
                ## loss2 for attr_task
                avg_loss2 += loss2.item()
                cur_loss2 = loss2.item()
                ## loss3 for clip
                avg_loss3 += loss3.item()
                cur_loss3 = loss3.item()


            post_fix = {
                "epoch": epoch,
                "rec_avg_loss": '{:.8f}'.format(rec_avg_loss / len(rec_data_iter)),
                "rec_cur_loss": '{:.8f}'.format(rec_cur_loss),
                "avg_loss1": '{:.8f}'.format(avg_loss1 / len(rec_data_iter)),
                "cur_loss1": '{:.8f}'.format(cur_loss1),
                "avg_loss2": '{:.8f}'.format(avg_loss2 / len(rec_data_iter)),
                "cur_loss2": '{:.8f}'.format(cur_loss2),
                "avg_loss3": '{:.8f}'.format(avg_loss3 / len(rec_data_iter)),
                "cur_loss3": '{:.8f}'.format(cur_loss3),
            }

            if verbose:
                self.args.writer.add_scalar('loss', rec_avg_loss / len(rec_data_iter), epoch)

            # if (epoch + 1) % self.args.log_freq == 0:
            #     print(str(post_fix))

            # with open(self.args.log_file, 'a') as f:
            #     f.write(str(post_fix) + '\n')

            return post_fix, rec_avg_loss / len(rec_data_iter)
        else:
            self.model.eval()

            pred_list = None

            if full_sort:
                answer_list = None
                for i, batch in rec_data_iter:
                    # 0. batch_data will be sent into the device(GPU or cpu)
                    batch = tuple(t.to(self.device) for t in batch)
                    user_ids, input_ids, target_pos, target_neg, answers, attrs, target_attr = batch
                    # recommend_output, attr_out,image_out, text_out = self.model.finetune(input_ids, attrs)

                    if self.args.Ours:
                        
                        if hasattr(self.args, 'MMOE'):
                            if self.args.MMOE:
                                recommend_output, attr_out, image_out, text_out = self.model.finetune(input_ids, attrs)
                            else:
                                recommend_output = self.model.finetune(input_ids, attrs)
                        else:
                            recommend_output = self.model.finetune(input_ids, attrs)
                    else:
                        recommend_output = self.model.finetune(input_ids, attrs)

                    #     if hasattr(self.args, 'MLP'):
                    #         if self.args.MLP:
                    #             recommend_output, attr_out, image_out, text_out = self.model.finetune(input_ids, attrs)
                    #         else:
                    #             recommend_output = self.model.finetune(input_ids, attrs)
                    #     else:
                    #         recommend_output = self.model.finetune(input_ids, attrs)
                    # else:
                    #     recommend_output = self.model.finetune(input_ids, attrs)

                    #     if hasattr(self.args, 'Trans'):
                    #         if self.args.Trans:
                    #             recommend_output, attr_out, image_out, text_out = self.model.finetune(input_ids, attrs)
                    #         else:
                    #             recommend_output = self.model.finetune(input_ids, attrs)
                    #     else:
                    #         recommend_output = self.model.finetune(input_ids, attrs)
                    # else:
                    #     recommend_output = self.model.finetune(input_ids, attrs)
                    


                    recommend_output = recommend_output[:, -1, :]
                    # 推荐的结果

                    rating_pred = self.predict_full(recommend_output)

                    rating_pred = rating_pred.cpu().data.numpy().copy()
                    batch_user_index = user_ids.cpu().numpy()
                    rating_pred[self.args.train_matrix[batch_user_index].toarray() > 0] = 0
                    # reference: https://stackoverflow.com/a/23734295, https://stackoverflow.com/a/20104162
                    # argpartition 时间复杂度O(n)  argsort O(nlogn) 只会做
                    # 加负号"-"表示取大的值
                    ind = np.argpartition(rating_pred, -50)[:, -50:]
                    # 根据返回的下标 从对应维度分别取对应的值 得到每行topk的子表
                    arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]
                    # 对子表进行排序 得到从大到小的顺序
                    arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]
                    # 再取一次 从ind中取回 原来的下标
                    batch_pred_list = ind[np.arange(len(rating_pred))[:, None], arr_ind_argsort]

                    if i == 0:
                        pred_list = batch_pred_list
                        answer_list = answers.cpu().data.numpy()
                    else:
                        pred_list = np.append(pred_list, batch_pred_list, axis=0)
                        answer_list = np.append(answer_list, answers.cpu().data.numpy(), axis=0)
                return self.get_full_sort_score(epoch, answer_list, pred_list, verbose)

            else:
                for i, batch in rec_data_iter:
                    # 0. batch_data will be sent into the device(GPU or cpu)
                    batch = tuple(t.to(self.device) for t in batch)
                    user_ids, input_ids, target_pos, target_neg, answers, sample_negs, attrs = batch
                    recommend_output = self.model.finetune(input_ids)
                    test_neg_items = torch.cat((answers, sample_negs), -1)
                    recommend_output = recommend_output[:, -1, :]

                    test_logits = self.predict_sample(recommend_output, test_neg_items)
                    test_logits = test_logits.cpu().detach().numpy().copy()
                    if i == 0:
                        pred_list = test_logits
                    else:
                        pred_list = np.append(pred_list, test_logits, axis=0)

                return self.get_sample_scores(epoch, pred_list, verbose)

    def get_metric(self, full_sort = False, sample_sort = False, verbose= False, epoch=None):
        self.model.eval()
        str_code = 'final'
        epoch = epoch if epoch is not None else 0
        rec_data_iter = tqdm.tqdm(enumerate(self.test_dataloader),
                            desc="Recommendation EP_%s" % (str_code),
                            total=len(self.test_dataloader),
                            bar_format="{l_bar}{r_bar}")
        
        full_res,sample_res = None,None

        pred_list = None
        if full_sort:
            answer_list = None
            for i, batch in rec_data_iter:
                # 0. batch_data will be sent into the device(GPU or cpu)
                batch = tuple(t.to(self.device) for t in batch)
                user_ids, input_ids, target_pos, target_neg, answers, sample_negs, attrs = batch
                recommend_output = self.model.finetune(input_ids)

                recommend_output = recommend_output[:, -1, :]
                # 推荐的结果

                rating_pred = self.predict_full(recommend_output)

                rating_pred = rating_pred.cpu().data.numpy().copy()
                batch_user_index = user_ids.cpu().numpy()
                rating_pred[self.args.train_matrix[batch_user_index].toarray() > 0] = 0
                # reference: https://stackoverflow.com/a/23734295, https://stackoverflow.com/a/20104162
                # argpartition 时间复杂度O(n)  argsort O(nlogn) 只会做
                # 加负号"-"表示取大的值
                ind = np.argpartition(rating_pred, -50)[:, -50:]
                # 根据返回的下标 从对应维度分别取对应的值 得到每行topk的子表
                arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]
                # 对子表进行排序 得到从大到小的顺序
                arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]
                # 再取一次 从ind中取回 原来的下标
                batch_pred_list = ind[np.arange(len(rating_pred))[:, None], arr_ind_argsort]

                if i == 0:
                    pred_list = batch_pred_list
                    answer_list = answers.cpu().data.numpy()
                else:
                    pred_list = np.append(pred_list, batch_pred_list, axis=0)
                    answer_list = np.append(answer_list, answers.cpu().data.numpy(), axis=0)
            # print(len(pred_list),len(answer_list))
            _, full_res = self.get_full_sort_score(epoch, answer_list, pred_list, verbose)

        if sample_sort:
            for i, batch in rec_data_iter:
                # 0. batch_data will be sent into the device(GPU or cpu)
                batch = tuple(t.to(self.device) for t in batch)
                user_ids, input_ids, target_pos, target_neg, answers, sample_negs, attrs = batch
                recommend_output = self.model.finetune(input_ids)
                test_neg_items = torch.cat((answers, sample_negs), -1)
                recommend_output = recommend_output[:, -1, :]

                test_logits = self.predict_sample(recommend_output, test_neg_items)
                test_logits = test_logits.cpu().detach().numpy().copy()
                if i == 0:
                    pred_list = test_logits
                else:
                    pred_list = np.append(pred_list, test_logits, axis=0)

            _, sample_res = self.get_sample_scores(epoch, pred_list, verbose)

        return sample_res, full_res