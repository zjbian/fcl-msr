import numpy as np
import tqdm
import random
import os
import torch
import torch.nn as nn
from torch.optim import Adam
import matplotlib.pyplot as plt
from datetime import datetime
from utils import recall_at_k, ndcg_k, get_metric
import csv
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

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

    def test(self, epoch, full_sort=False, verbose = False,collect_embeddings=False):

        # if hasattr(self.args, 'gap'):
        #     if self.args.gap != 0:
        #         user_seq = self.test_dataloader.dataset.user_seq
        #         from utils import generate_rating_matrix_test
        #         self.args.train_matrix = generate_rating_matrix_test(user_seq, len(user_seq), self.args.item_size)
        if collect_embeddings:
            metrics, result_info, collected_embeddings = self.iteration(epoch, self.test_dataloader, full_sort, train=False, verbose=verbose, collect_embeddings=collect_embeddings)
            return metrics, result_info, collected_embeddings
        else:
            return self.iteration(epoch, self.test_dataloader, full_sort, train=False, verbose=verbose)
        

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
        self.best_alpha_save_path = os.path.join(self.args.save_path, "best_model_alpha.csv")
        self.plot_save_path = self.args.save_path
        self.epochs_record = []
        self.alpha_rec_record = []
        self.alpha_attr_record = []
        self.alpha_clip_record = []
        self.loss1_record = []    # recommendation loss
        self.loss2_record = []    # attribute loss
        self.loss3_record = []    # CLIP loss
        self.ndcg10_record = []   # NDCG@10
        self.recall10_record = [] # Recall@10
        self.recall50_record = [] # Recall@50
        self.ndcg50_record = []   # NDCG@50
    def _compute_best_model_alpha(self, dataloader):
        """
        Compute stable task weights alpha using a small validation subset.
        This runs independently of the training loop.
        """
        self.model.eval()
        alpha_list = []
        max_batches = 5  # Use a small number of batches for efficiency
        batch_count = 0

        with torch.no_grad():
            for batch in dataloader:
                if batch_count >= max_batches:
                    break

                # Step 1: Extract input_ids and attrs from batch
                # Batch structure: (user_ids, input_ids, target_pos, target_neg, attrs, target_attr, ...)
                try:
                    input_ids = batch[1].to(self.device)
                    attrs = batch[4].to(self.device)
                except (IndexError, TypeError) as e:
                    print(f"Warning: cannot extract input_ids/attrs from batch. Skipping. Error: {e}")
                    continue

                # Step 2: Forward pass
                try:
                    sequence_output, attr_out, image_out, text_out, _ = self.model.finetune(input_ids, attrs)
                except Exception as e:
                    print(f"Warning: forward pass failed. Skipping batch. Error: {e}")
                    continue

                # Step 3: Compute per-task losses
                try:
                    target_pos = batch[2].to(self.device)
                    loss1 = self.cce_loss(sequence_output, target_pos)

                    target_attr = batch[5].to(self.device)
                    loss2 = self.attr_loss(attr_out, target_attr)

                    loss3 = 0.1 * self.model.clip_pretrain_loss(image_out, text_out, target_pos)
                except Exception as e:
                    print(f"Warning: loss computation failed. Skipping batch. Error: {e}")
                    continue

                # Step 4: Collect task-specific gradients on shared parameters
                try:
                    with torch.enable_grad():
                        task_losses, task_shared_grads, _ = self.model.compute_task_losses_and_grads(
                            sequence_output.detach(),
                            attr_out.detach(),
                            image_out.detach(),
                            text_out.detach(),
                            target_pos,
                            target_attr,
                            [loss1, loss2, loss3]
                        )
                except Exception as e:
                    print(f"Warning: gradient collection failed. Skipping batch. Error: {e}")
                    continue

                # Step 5: Filter invalid gradients and solve for alpha
                task_shared_grads = [g for g in task_shared_grads if g is not None and g.numel() > 0]
                if len(task_shared_grads) == 3:
                    alpha = self.model.frank_wolfe_solver(task_shared_grads)
                    alpha_list.append(alpha.cpu().numpy())
                    print(f"  Batch {batch_count+1} alpha: {alpha.cpu().numpy().round(4)}")

                batch_count += 1

        # Step 6: Average and normalize alpha across batches
        if not alpha_list:
            print("Warning: no valid alpha computed. Using uniform weights [0.333, 0.333, 0.333].")
            return np.array([0.333, 0.333, 0.333])

        best_alpha = np.mean(alpha_list, axis=0)
        best_alpha = best_alpha / best_alpha.sum()
        return best_alpha.round(4)
    def _get_model_shared_params(self):
        """Get shared parameters from the model."""
        if hasattr(self.model, 'get_shared_params'):
            return self.model.get_shared_params()
        return []

    def _get_model_task_specific_params(self):
        """Get task-specific parameters from the model."""
        if hasattr(self.model, 'get_task_specific_params'):
            return self.model.get_task_specific_params()
        return []

    def iteration(self, epoch, dataloader, full_sort=False, train=True, verbose = False, collect_embeddings=False):

        str_code = "train" if train else "test"

        # Setting the tqdm progress bar

        rec_data_iter = tqdm.tqdm(enumerate(dataloader),
                                  desc="Recommendation EP_%s:%d" % (str_code, epoch),
                                  total=len(dataloader),
                                  bar_format="{l_bar}{r_bar}")

        collected_embeddings = None
        if collect_embeddings and not train:
            self.fusion_method = self.get_fusion_method()
            # Collect fused global features
            collected_embeddings = {
                'method': self.fusion_method,
                'ID': [], 'Attr': [], 'Img': [], 'Text': [], 'Fused': []
            }


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

            batch_alphas = []

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

                finetune_called = False  # Ensure finetune is called only once
                if self.args.Ours:
                    if self.args.MMOE and not finetune_called:
                        print("hasattr--->MMOE",self.args.MMOE)
                        sequence_output, attr_out, image_out, text_out,_ = self.model.finetune(input_ids, attrs)
                        finetune_called = True
                    elif self.args.MLP and not finetune_called:
                        # print("hasattr--->MLP",self.args.MLP)
                        print("hasattr--->MLP")
                        sequence_output, attr_out, image_out, text_out,_ = self.model.finetune(input_ids, attrs)
                        # print(sequence_output.shape)
                        finetune_called = True
                    elif self.args.Trans and not finetune_called:
                        print("hasattr--->Transformer")
                        # print(input_ids.shape)
                        sequence_output, attr_out, image_out, text_out,_ = self.model.finetune(input_ids, attrs)
                        # print(sequence_output.shape)
                        finetune_called = True
                    elif not finetune_called:
                        sequence_output = self.model.finetune(input_ids, attrs)
                else:
                    sequence_output = self.model.finetune(input_ids, attrs)              


                
                ## CCEloss---nip
                loss1 = self.cce_loss(sequence_output, target_pos)            
                loss3 = 0.1*self.model.clip_pretrain_loss(image_out, text_out, target_pos)
                loss2 = self.attr_loss(attr_out, target_attr)
                loss = loss1 + loss3 + loss2

                loss_lst = [loss1, loss2, loss3]
                alpha = None
                if hasattr(self.model, 'compute_task_losses_and_grads') and hasattr(self.model, 'frank_wolfe_solver') and self.args.auto_weight:
                    # Multi-task gradient balancing via Frank-Wolfe
                    task_losses, task_shared_grads, shared_params = self.model.compute_task_losses_and_grads(
                        sequence_output, attr_out, image_out, text_out, target_pos, target_attr, loss_lst
                    )

                    # Filter invalid gradients
                    task_shared_grads = [g for g in task_shared_grads if g is not None and g.numel() > 0]
                    if len(task_shared_grads) != len(task_losses):
                        # Fallback to standard update
                        self.optim.zero_grad()
                        loss.backward()
                        self.optim.step()
                        print(f"Warning: Batch {i} gradient collection failed, using standard update")
                    else:
                        # Frank-Wolfe solver for gradient weights (Algorithm 2)
                        alpha = self.model.frank_wolfe_solver(task_shared_grads)
                        if i % 50 == 0:
                            print(f"Batch {i} - alpha: rec={alpha[0]:.3f}, attr={alpha[1]:.3f}, clip={alpha[2]:.3f}")

                        self.optim.zero_grad()

                        # Shared params: weighted gradient update
                        shared_loss = alpha[0] * task_losses['rec'] + alpha[1] * task_losses['attr'] + alpha[2] * task_losses['clip']
                        shared_loss.backward(retain_graph=True)

                        # Task-specific params: independent gradient update
                        task_specific_params = self._get_model_task_specific_params()
                        if task_specific_params:
                            task_specific_loss = loss1 + loss2 + loss3
                            task_specific_loss.backward()

                        self.optim.step()
                else:
                    # Standard single-task update
                    self.optim.zero_grad()
                    loss.backward()
                    self.optim.step()
                if alpha is not None:
                    batch_alphas.append(alpha.cpu().numpy())
                
    

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
            self.loss1_record.append(avg_loss1 / len(rec_data_iter))
            self.loss2_record.append(avg_loss2 / len(rec_data_iter))
            self.loss3_record.append(avg_loss3 / len(rec_data_iter))
            self.epochs_record.append(epoch)
            if batch_alphas:
                avg_alpha = np.mean(batch_alphas, axis=0)
                self.alpha_rec_record.append(avg_alpha[0])
                self.alpha_attr_record.append(avg_alpha[1])
                self.alpha_clip_record.append(avg_alpha[2])
            else:
                print("Warning: no alpha weights recorded this epoch, possibly due to gradient collection failure.")
                # Fill with zeros if no weights recorded
                self.alpha_rec_record.append(0)
                self.alpha_attr_record.append(0)
                self.alpha_clip_record.append(0)
            post_fix = {
                "epoch": epoch,
                "rec_avg_loss": '{:.8f}'.format(rec_avg_loss / len(rec_data_iter)),
                "rec_cur_loss": '{:.8f}'.format(rec_cur_loss),
                "avg_loss1": '{:.8f}'.format(avg_loss1 / len(rec_data_iter)),
                "cur_loss1": '{:.8f}'.format(cur_loss1),
                "avg_loss2": '{:.8f}'.format(avg_loss2 / len(rec_data_iter)),
                "cur_loss2": '{:.8f}'.format(cur_loss2),
                "avg_loss3": '{:.8f}'.format(avg_loss3 / len(rec_data_iter)),
                "cur_loss3": '{:.8f}'.format(cur_loss3)
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
                        if self.args.MMOE:
                            recommend_output, attr_out, image_out, text_out, original_embeddings = self.model.finetune(input_ids, attrs)
                    # MLP fusion branch
                        elif self.args.MLP:
                            recommend_output, attr_out, image_out, text_out, original_embeddings = self.model.finetune(input_ids, attrs)
                    # Transformer fusion branch
                        elif self.args.Trans:
                            recommend_output, attr_out, image_out, text_out, original_embeddings = self.model.finetune(input_ids, attrs)
                        else:
                        # Try to unpack 5 values; fallback to single output
                            try:
                                recommend_output, attr_out, image_out, text_out, original_embeddings = self.model.finetune(input_ids, attrs)
                            except:
                                recommend_output = self.model.finetune(input_ids, attrs)
                    else:
                        try:
                            recommend_output, attr_out, image_out, text_out, original_embeddings = self.model.finetune(input_ids, attrs)
                        except:
                            recommend_output = self.model.finetune(input_ids, attrs)

                    
                    if collect_embeddings and original_embeddings is not None:
                        for modal in ['ID', 'Attr', 'Img', 'Text', 'Fused']:
                            if modal in original_embeddings and original_embeddings[modal] is not None:
                                if len(original_embeddings[modal].shape) == 2 and original_embeddings[modal].shape[0] > 0:
                                    collected_embeddings[modal].append(original_embeddings[modal])

                    recommend_output = recommend_output[:, -1, :]

                    rating_pred = self.predict_full(recommend_output)

                    rating_pred = rating_pred.cpu().data.numpy().copy()
                    batch_user_index = user_ids.cpu().numpy()
                    rating_pred[self.args.train_matrix[batch_user_index].toarray() > 0] = 0
                    # argpartition O(n) is faster than argsort O(n log n) for top-k selection
                    ind = np.argpartition(rating_pred, -50)[:, -50:]
                    arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]
                    arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]
                    batch_pred_list = ind[np.arange(len(rating_pred))[:, None], arr_ind_argsort]

                    if i == 0:
                        pred_list = batch_pred_list
                        answer_list = answers.cpu().data.numpy()
                    else:
                        pred_list = np.append(pred_list, batch_pred_list, axis=0)
                        answer_list = np.append(answer_list, answers.cpu().data.numpy(), axis=0)
                if collect_embeddings:
                    # Merge embeddings across batches, filtering empty modalities
                    for modal in ['ID', 'Attr', 'Img', 'Text', 'Fused']:
                        if len(collected_embeddings[modal]) > 0:
                            collected_embeddings[modal] = np.vstack(collected_embeddings[modal])
                        else:
                            del collected_embeddings[modal]
                    metrics, result_info = self.get_full_sort_score(epoch, answer_list, pred_list, verbose)
                    self.recall10_record.append(metrics[2])
                    self.recall50_record.append(metrics[6])
                    self.ndcg10_record.append(metrics[3])
                    self.ndcg50_record.append(metrics[7])

                    return metrics, result_info, collected_embeddings
                else:
                    metrics, result_info = self.get_full_sort_score(epoch, answer_list, pred_list, verbose)
                    self.recall10_record.append(metrics[2])
                    self.recall50_record.append(metrics[6])
                    self.ndcg10_record.append(metrics[3])
                    self.ndcg50_record.append(metrics[7])
                    return self.get_full_sort_score(epoch, answer_list, pred_list, verbose)
                

            else:
                for i, batch in rec_data_iter:
                    # 0. batch_data will be sent into the device(GPU or cpu)
                    batch = tuple(t.to(self.device) for t in batch)
                    user_ids, input_ids, target_pos, target_neg, answers, sample_negs, attrs = batch

                    
                    
                    if self.args.Ours:
                            if self.args.MMOE:
                                recommend_output, attr_out, image_out, text_out, original_embeddings = self.model.finetune(input_ids, attrs)
                            elif self.args.MLP:
                                recommend_output, attr_out, image_out, text_out, original_embeddings = self.model.finetune(input_ids, attrs)
                            elif self.args.Trans:
                                recommend_output, attr_out, image_out, text_out, original_embeddings = self.model.finetune(input_ids, attrs)
                            else:
                                try:
                                    recommend_output, attr_out, image_out, text_out, original_embeddings = self.model.finetune(input_ids, attrs)
                                except:
                                    recommend_output = self.model.finetune(input_ids, attrs)
                    else:
                        try:
                            recommend_output, attr_out, image_out, text_out, original_embeddings = self.model.finetune(input_ids, attrs)
                        except:
                            recommend_output = self.model.finetune(input_ids, attrs)

                    if collect_embeddings and original_embeddings is not None:
                        for modal in ['ID', 'Attr', 'Img', 'Text', 'Fused']:
                            if modal in original_embeddings and original_embeddings[modal] is not None:
                                if len(original_embeddings[modal].shape) == 2 and original_embeddings[modal].shape[0] > 0:
                                    collected_embeddings[modal].append(original_embeddings[modal])
                    
                    test_neg_items = torch.cat((answers, sample_negs), -1)
                    recommend_output = recommend_output[:, -1, :]

                    test_logits = self.predict_sample(recommend_output, test_neg_items)
                    test_logits = test_logits.cpu().detach().numpy().copy()
                    
                    if i == 0:
                        pred_list = test_logits
                    else:
                        pred_list = np.append(pred_list, test_logits, axis=0)
                if collect_embeddings:
                    # Merge embeddings across batches
                    for modal in ['ID', 'Attr', 'Img', 'Text', 'Fused']:
                        if len(collected_embeddings[modal]) > 0:
                            collected_embeddings[modal] = np.vstack(collected_embeddings[modal])
                        else:
                            del collected_embeddings[modal]
                    metrics, result_info = self.get_sample_scores(epoch, pred_list, verbose)
                    return metrics, result_info, collected_embeddings
                else:
                    return self.get_sample_scores(epoch, pred_list, verbose)

    def plot_final_losses(self):
        """Plot and save loss curves after training completes."""
        if not self.epochs_record:
            print("Warning: no training data recorded, cannot generate plots.")
            return

        print("\nTraining complete. Generating loss plots and saving data...")

        # Save loss data to CSV
        data_filename = f"{self.args.data_name}_loss_no_auto.csv"
        data_save_path = os.path.join(self.plot_save_path, data_filename)
        os.makedirs(self.plot_save_path, exist_ok=True)
        try:
            with open(data_save_path, mode='w', newline='') as csv_file:
                fieldnames = ['Epoch', 'Loss1_Rec', 'Loss2_Attr', 'Loss3_Clip']
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writeheader()
                for i in range(len(self.epochs_record)):
                    writer.writerow({
                        'Epoch': self.epochs_record[i],
                        'Loss1_Rec': self.loss1_record[i] if i < len(self.loss1_record) else None,
                        'Loss2_Attr': self.loss2_record[i] if i < len(self.loss2_record) else None,
                        'Loss3_Clip': self.loss3_record[i] if i < len(self.loss3_record) else None,
                    })
            print(f"Loss data saved to: {data_save_path}")
        except Exception as e:
            print(f"Warning: failed to save loss data: {e}")

        # Global font settings
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['font.size'] = 14

        # Prepare Y-axis range
        all_losses = []
        if self.loss1_record:
            all_losses.extend(self.loss1_record)
        if self.loss2_record:
            all_losses.extend(self.loss2_record)
        if self.loss3_record:
            all_losses.extend(self.loss3_record)

        if not all_losses:
            print("Warning: no valid loss data, cannot generate plots.")
            return

        y_min, y_max = min(all_losses), max(all_losses)
        y_margin = (y_max - y_min) * 0.1
        y_lim = [y_min - y_margin, y_max + y_margin]
        y_ticks = np.linspace(y_lim[0], y_lim[1], 6)

        # Plot 1: Loss curves for all epochs
        fig1, ax1 = plt.subplots(figsize=(16, 8))
        if self.loss1_record:
            ax1.plot(self.epochs_record, self.loss1_record, 'r-', label='Reconstruction Loss (Loss1)',
                    linewidth=3, marker='o', markersize=6)
        if self.loss2_record:
            ax1.plot(self.epochs_record, self.loss2_record, 'g-', label='Attribute Loss (Loss2)',
                    linewidth=3, marker='s', markersize=6)
        if self.loss3_record:
            ax1.plot(self.epochs_record, self.loss3_record, 'b-', label='CLIP Loss (Loss3)',
                    linewidth=3, marker='^', markersize=6)

        ax1.set_xlabel('Epoch', fontsize=20)
        ax1.set_ylabel('Loss Value', fontsize=20)
        ax1.legend(loc='best', fontsize=18)
        ax1.set_ylim(y_lim)
        ax1.set_yticks(y_ticks)
        ax1.set_xlim(min(self.epochs_record), max(self.epochs_record))
        ax1.set_xticks(range(min(self.epochs_record), max(self.epochs_record) + 1, 5))
        ax1.tick_params(axis='both', which='major', labelsize=16)

        plot1_save_path = os.path.join(self.plot_save_path, f"{self.args.data_name}_losses_all_epochs.png")
        plt.tight_layout()
        plt.savefig(plot1_save_path, dpi=300, bbox_inches='tight')
        plt.close(fig1)
        print(f"Plot 1 (all epochs) saved to: {plot1_save_path}")

        # Plot 2: Sparse loss curves (every N epochs)
        step = 8 if len(self.epochs_record) > 30 else 5
        filtered_epochs = self.epochs_record[::step]
        filtered_loss1 = self.loss1_record[::step] if self.loss1_record else []
        filtered_loss2 = self.loss2_record[::step] if self.loss2_record else []
        filtered_loss3 = self.loss3_record[::step] if self.loss3_record else []

        # Ensure last epoch is included
        if filtered_epochs and filtered_epochs[-1] != self.epochs_record[-1]:
            filtered_epochs.append(self.epochs_record[-1])
            if self.loss1_record: filtered_loss1.append(self.loss1_record[-1])
            if self.loss2_record: filtered_loss2.append(self.loss2_record[-1])
            if self.loss3_record: filtered_loss3.append(self.loss3_record[-1])

        fig2, ax2 = plt.subplots(figsize=(16, 8))
        if filtered_loss1:
            ax2.plot(filtered_epochs, filtered_loss1, 'r-', label='Reconstruction Loss (Loss1)',
                    linewidth=3, marker='o', markersize=10)
        if filtered_loss2:
            ax2.plot(filtered_epochs, filtered_loss2, 'g-', label='Attribute Loss (Loss2)',
                    linewidth=3, marker='s', markersize=10)
        if filtered_loss3:
            ax2.plot(filtered_epochs, filtered_loss3, 'b-', label='CLIP Loss (Loss3)',
                    linewidth=3, marker='^', markersize=10)

        ax2.set_xlabel('Epoch', fontsize=20)
        ax2.set_ylabel('Loss Value', fontsize=20)
        ax2.legend(loc='best', fontsize=18)
        ax2.set_ylim(y_lim)
        ax2.set_yticks(y_ticks)
        ax2.set_xlim(min(filtered_epochs), max(filtered_epochs))
        ax2.set_xticks(range(min(filtered_epochs), max(filtered_epochs) + 1, 5))
        ax2.tick_params(axis='both', which='major', labelsize=16)

        plot2_save_path = os.path.join(self.plot_save_path, f"{self.args.data_name}_losses_every_{step}_epochs.png")
        plt.tight_layout()
        plt.savefig(plot2_save_path, dpi=300, bbox_inches='tight')
        plt.close(fig2)
        print(f"Plot 2 (every {step} epochs) saved to: {plot2_save_path}")

        plt.rcParams.update(plt.rcParamsDefault)
        print("Loss plots and data saved.")

    def plot_final_results(self):
        """Plot and save weight curves after training completes."""
        if not self.epochs_record:
            print("Warning: no training data recorded, cannot generate plots.")
            return

        print("\nTraining complete. Generating weight plots and saving data...")

        # Save weight and loss data to CSV
        data_filename = f"{self.args.data_name}_training_data.csv"
        data_save_path = os.path.join(self.plot_save_path, data_filename)
        os.makedirs(self.plot_save_path, exist_ok=True)
        try:
            with open(data_save_path, mode='w', newline='') as csv_file:
                fieldnames = ['Epoch', 'Alpha_Rec', 'Alpha_Attr', 'Alpha_Clip', 'Loss1_Rec', 'Loss2_Attr', 'Loss3_Clip']
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writeheader()
                for i in range(len(self.epochs_record)):
                    writer.writerow({
                        'Epoch': self.epochs_record[i],
                        'Alpha_Rec': self.alpha_rec_record[i],
                        'Alpha_Attr': self.alpha_attr_record[i],
                        'Alpha_Clip': self.alpha_clip_record[i],
                        'Loss1_Rec': self.loss1_record[i] if i < len(self.loss1_record) else None,
                        'Loss2_Attr': self.loss2_record[i] if i < len(self.loss2_record) else None,
                        'Loss3_Clip': self.loss3_record[i] if i < len(self.loss3_record) else None,
                    })
            print(f"Training data saved to: {data_save_path}")
        except Exception as e:
            print(f"Warning: failed to save training data: {e}")

        # Global font settings
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['font.size'] = 14

        # Prepare Y-axis range
        all_weights = self.alpha_rec_record + self.alpha_attr_record + self.alpha_clip_record
        y_min, y_max = min(all_weights), max(all_weights)
        y_margin = (y_max - y_min) * 0.1
        y_lim = [y_min - y_margin, y_max + y_margin]
        y_ticks = np.linspace(y_lim[0], y_lim[1], 6)

        # Plot 1: Weight curves for all epochs
        fig1, ax1 = plt.subplots(figsize=(16, 8))
        ax1.plot(self.epochs_record, self.alpha_rec_record, 'r-', label='Recommendation Weight',
                linewidth=3, marker='o', markersize=6)
        ax1.plot(self.epochs_record, self.alpha_attr_record, 'g-', label='Attribute Weight',
                linewidth=3, marker='s', markersize=6)
        ax1.plot(self.epochs_record, self.alpha_clip_record, 'b-', label='CLIP Weight',
                linewidth=3, marker='^', markersize=6)

        ax1.set_xlabel('Epoch', fontsize=20)
        ax1.set_ylabel('Weight Value', fontsize=20)
        ax1.legend(loc='best', fontsize=18)
        ax1.set_ylim(y_lim)
        ax1.set_yticks(y_ticks)
        ax1.set_xlim(min(self.epochs_record), max(self.epochs_record))
        ax1.set_xticks(range(min(self.epochs_record), max(self.epochs_record) + 1, 5))
        ax1.tick_params(axis='both', which='major', labelsize=16)

        plot1_save_path = os.path.join(self.plot_save_path, f"{self.args.data_name}_weights_all_epochs.png")
        plt.tight_layout()
        plt.savefig(plot1_save_path, dpi=300, bbox_inches='tight')
        plt.close(fig1)
        print(f"Plot 1 (all epochs) saved to: {plot1_save_path}")

        # Plot 2: Sparse weight curves (every N epochs)
        step = 8 if len(self.epochs_record) > 30 else 5
        filtered_epochs = self.epochs_record[::step]
        filtered_alpha_rec = self.alpha_rec_record[::step]
        filtered_alpha_attr = self.alpha_attr_record[::step]
        filtered_alpha_clip = self.alpha_clip_record[::step]

        if filtered_epochs and filtered_epochs[-1] != self.epochs_record[-1]:
            filtered_epochs.append(self.epochs_record[-1])
            filtered_alpha_rec.append(self.alpha_rec_record[-1])
            filtered_alpha_attr.append(self.alpha_attr_record[-1])
            filtered_alpha_clip.append(self.alpha_clip_record[-1])

        fig2, ax2 = plt.subplots(figsize=(16, 8))
        ax2.plot(filtered_epochs, filtered_alpha_rec, 'r-', label='Recommendation Weight',
                linewidth=3, marker='o', markersize=10)
        ax2.plot(filtered_epochs, filtered_alpha_attr, 'g-', label='Attribute Weight',
                linewidth=3, marker='s', markersize=10)
        ax2.plot(filtered_epochs, filtered_alpha_clip, 'b-', label='CLIP Weight',
                linewidth=3, marker='^', markersize=10)

        ax2.set_xlabel('Epoch', fontsize=20)
        ax2.set_ylabel('Weight Value', fontsize=20)
        ax2.legend(loc='best', fontsize=18)
        ax2.set_ylim(y_lim)
        ax2.set_yticks(y_ticks)
        ax2.set_xlim(min(filtered_epochs), max(filtered_epochs))
        ax2.set_xticks(range(min(filtered_epochs), max(filtered_epochs) + 1, 5))
        ax2.tick_params(axis='both', which='major', labelsize=16)

        plot2_save_path = os.path.join(self.plot_save_path, f"{self.args.data_name}_weights_every_{step}_epochs.png")
        plt.tight_layout()
        plt.savefig(plot2_save_path, dpi=300, bbox_inches='tight')
        plt.close(fig2)
        print(f"Plot 2 (every {step} epochs) saved to: {plot2_save_path}")

        plt.rcParams.update(plt.rcParamsDefault)
        print("Weight plots and data saved.")
    def get_fusion_method(self):
        """Get the fusion method type."""
        if self.args.MMOE:
            return "MMOE"
        elif self.args.MLP:
            return "MLP"
        elif self.args.Trans:
            return "Transformer"
        else:
            return "Baseline"

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

                rating_pred = self.predict_full(recommend_output)

                rating_pred = rating_pred.cpu().data.numpy().copy()
                batch_user_index = user_ids.cpu().numpy()
                rating_pred[self.args.train_matrix[batch_user_index].toarray() > 0] = 0
                # argpartition O(n) faster than argsort O(n log n) for top-k
                ind = np.argpartition(rating_pred, -50)[:, -50:]
                arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]
                arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]
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