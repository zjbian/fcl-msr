import random

import torch
from torch.utils.data import Dataset

from utils import neg_sample

class SASRecDataset(Dataset):

    def __init__(self, args, user_seq, test_neg_items=None, data_type='train'):
        self.args = args
        self.user_seq = user_seq
        self.test_neg_items = test_neg_items
        self.data_type = data_type
        self.max_len = args.max_seq_length
        self.item2attr = args.item2attribute
        self.item2attr['0'] = [0]*10

    def __getitem__(self, index):

        user_id = index
        items = self.user_seq[index]

        assert self.data_type in {"train", "valid", "test"}

        # [0, 1, 2, 3, 4, 5, 6]
        # train [0, 1, 2, 3]
        # target [1, 2, 3, 4]

        # valid [0, 1, 2, 3, 4]
        # answer [5]

        # test [0, 1, 2, 3, 4, 5]
        # answer [6]
        if self.data_type == "train":
            input_ids = items[:-3]
            target_pos = items[1:-2]
            answer = [0] # no use

        elif self.data_type == 'valid':
            input_ids = items[:-2]
            target_pos = items[1:-1]
            answer = [items[-2]]

        else:
            input_ids = items[:-1]
            target_pos = items[1:]
            answer = [items[-1]]


        target_neg = []
        seq_set = set(items)
        for _ in input_ids:
            target_neg.append(neg_sample(seq_set, self.args.item_size))

        pad_len = self.max_len - len(input_ids)
        input_ids = [0] * pad_len + input_ids
        target_pos = [0] * pad_len + target_pos
        target_neg = [0] * pad_len + target_neg

        input_ids = input_ids[-self.max_len:]
        target_pos = target_pos[-self.max_len:]
        target_neg = target_neg[-self.max_len:]

        assert len(input_ids) == self.max_len
        assert len(target_pos) == self.max_len
        assert len(target_neg) == self.max_len

        attrs = []
        for id in input_ids:
            tmp = self.item2attr[str(id)]
            if len(tmp) >= 10:
                attrs.append(tmp[:10])
            else:
                attrs.append(tmp + [0]*(10-len(tmp)))

        target_attr = []
        for id in target_pos:
            tmp = self.item2attr[str(id)]
            if len(tmp) >= 10:
                target_attr.append(tmp[:10])
            else:
                target_attr.append(tmp + [0]*(10-len(tmp)))


        if self.test_neg_items is not None:
            test_samples = self.test_neg_items[index]

            cur_tensors = (
                torch.tensor(user_id, dtype=torch.long), # user_id for testing
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(target_pos, dtype=torch.long),
                torch.tensor(target_neg, dtype=torch.long),
                torch.tensor(answer, dtype=torch.long),
                torch.tensor(test_samples, dtype=torch.long),
                torch.tensor(attrs, dtype=torch.long),
                torch.tensor(target_attr, dtype=torch.long),
            )
        else:
            cur_tensors = (
                torch.tensor(user_id, dtype=torch.long),  # user_id for testing
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(target_pos, dtype=torch.long),
                torch.tensor(target_neg, dtype=torch.long),
                torch.tensor(answer, dtype=torch.long),
                torch.tensor(attrs, dtype=torch.long),
                torch.tensor(target_attr, dtype=torch.long),
            )

        return cur_tensors

    def __len__(self):
        return len(self.user_seq)




class OODDataset(Dataset):
    def __init__(self, args, user_seq, test_neg_items=None, data_type='train'):
        self.args = args
        self.user_seq = user_seq
        self.test_neg_items = test_neg_items
        self.data_type = data_type
        self.max_len = args.max_seq_length
        self.item2attr = args.item2attribute
        self.item2attr['0'] = [0]*10
        if hasattr(self.args, 'gap'):
            self.gap = self.args.gap
        else:
            self.gap = 0
        
        tmp = []
        for seq in self.user_seq:
            if len(seq) > self.gap + 2:
                tmp.append(seq)
        self.user_seq = tmp

        # if data_type == 'test':
        #     tmp = []
        #     for seq in self.user_seq:
        #         for i in range(self.gap):
        #             tmp.append(seq[:-self.gap-1])
        #     self.user_seq = tmp

    def __getitem__(self, index):

        user_id = index
        items = self.user_seq[index]

        assert self.data_type in {"train", "valid", "test"}

        # [0, 1, 2, 3, 4, 5, 6]
        # train [0, 1, 2, 3]
        # target [1, 2, 3, 4]

        # valid [0, 1, 2, 3, 4]
        # answer [5]

        # test [0, 1, 2, 3, 4, 5]
        # answer [6]
        if self.data_type == "train":
            input_ids = items[:-3-self.gap]
            target_pos = items[1:-2-self.gap]
            answer = [0] # no use

        elif self.data_type == 'valid':
            input_ids = items[:-2-self.gap]
            target_pos = items[1:-1-self.gap]
            answer = [items[-2]]

        else:
            input_ids = items[:-1]
            target_pos = items[1:]
            answer = [items[-1]]


        target_neg = []
        seq_set = set(items)
        for _ in input_ids:
            target_neg.append(neg_sample(seq_set, self.args.item_size))

        pad_len = self.max_len - len(input_ids)
        input_ids = [0] * pad_len + input_ids
        target_pos = [0] * pad_len + target_pos
        target_neg = [0] * pad_len + target_neg

        input_ids = input_ids[-self.max_len:]
        target_pos = target_pos[-self.max_len:]
        target_neg = target_neg[-self.max_len:]

        assert len(input_ids) == self.max_len
        assert len(target_pos) == self.max_len
        assert len(target_neg) == self.max_len

        attrs = []
        for id in input_ids:
            tmp = self.item2attr[str(id)]
            if len(tmp) >= 10:
                attrs.append(tmp[:10])
            else:
                attrs.append(tmp + [0]*(10-len(tmp)))

        target_attr = []
        for id in target_pos:
            tmp = self.item2attr[str(id)]
            if len(tmp) >= 10:
                target_attr.append(tmp[:10])
            else:
                target_attr.append(tmp + [0]*(10-len(tmp)))


        if self.test_neg_items is not None:
            test_samples = self.test_neg_items[index]

            cur_tensors = (
                torch.tensor(user_id, dtype=torch.long), # user_id for testing
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(target_pos, dtype=torch.long),
                torch.tensor(target_neg, dtype=torch.long),
                torch.tensor(answer, dtype=torch.long),
                torch.tensor(test_samples, dtype=torch.long),
                torch.tensor(attrs, dtype=torch.long),
                torch.tensor(target_attr, dtype=torch.long),
            )
        else:
            cur_tensors = (
                torch.tensor(user_id, dtype=torch.long),  # user_id for testing
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(target_pos, dtype=torch.long),
                torch.tensor(target_neg, dtype=torch.long),
                torch.tensor(answer, dtype=torch.long),
                torch.tensor(attrs, dtype=torch.long),
                torch.tensor(target_attr, dtype=torch.long),
            )

        return cur_tensors

    def __len__(self):
        return len(self.user_seq)