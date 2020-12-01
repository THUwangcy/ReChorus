# -*- coding: UTF-8 -*-

import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List
from torch.nn.utils.rnn import pad_sequence

from models.BaseModel import SequentialModel
from models.BaseModel import GeneralModel
from utils import layers


class ContrastRecBeta(SequentialModel):
    extra_log_args = ['stage', 'emb_size', 'num_layers', 'num_heads', 'reorder_ratio', 'temperature', 'encoder']

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--stage', type=int, default=1,
                            help='Stage of training: 0-augmentation, 1-representation, 2-recommendation.')
        parser.add_argument('--emb_size', type=int, default=64,
                            help='Size of embedding vectors.')
        parser.add_argument('--num_layers', type=int, default=2,
                            help='Number of self-attention layers.')
        parser.add_argument('--num_heads', type=int, default=2,
                            help='Number of attention heads.')
        parser.add_argument('--reorder_ratio', type=float, default=0.7,
                            help='Ratio of historical sequence to be reordered.')
        parser.add_argument('--temperature', type=float, default=0.2,
                            help='Temperature in contrastive loss.')
        parser.add_argument('--future_window', type=int, default=5,
                            help='Use the subsequent future_window items to construct soft labels.')
        parser.add_argument('--encoder', type=str, default='SASRec',
                            help='Choose a sequence encoder: GRU4Rec, SASRec.')
        parser.add_argument('--checkpoint', type=str, default='',
                            help='Choose a pre-train model checkpoint.')
        return SequentialModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        self.stage = args.stage
        self.emb_size = args.emb_size
        self.max_his = args.history_max
        self.num_layers = args.num_layers
        self.num_heads = args.num_heads
        self.reorder_ratio = args.reorder_ratio
        self.temperature = args.temperature
        self.future_window = args.future_window
        self.encoder_name = args.encoder
        super().__init__(args, corpus)

        if self.stage == 1:
            if args.checkpoint == '':
                self.pre_path = '../model/ContrastRecBeta/Pre__{}__{}__encoder={}__temp={}__bsz={}.pt'.format(
                    corpus.dataset, args.random_seed, self.encoder_name, self.temperature, args.batch_size)
            self.model_path = self.pre_path
        else:
            self.pre_path = args.checkpoint

    def actions_before_train(self):
        if self.stage == 2:
            if os.path.exists(self.pre_path):
                self.load_model(self.pre_path)
            else:
                logging.warning('Train from scratch because pre-train model does not exist!')

    def _define_params(self):
        self.i_embeddings = nn.Embedding(self.item_num, self.emb_size)
        if self.encoder_name == 'GRU4Rec':
            self.encoder = GRU4RecEncoder(self.emb_size, self.emb_size)
        elif self.encoder_name == 'SASRec':
            self.encoder = SASRecEncoder(
                self.emb_size, self.num_layers, self.num_heads,self.max_his, self.dropout, self.device)
        else:
            raise ValueError('Invalid sequence encoder.')
        self.criterion = SupConLoss(self.device, temperature=self.temperature)

    def forward(self, feed_dict):
        self.check_list = []
        i_ids = feed_dict['item_id']  # [batch_size, -1]
        history = feed_dict['history_items']  # [batch_size, history_max]
        lengths = feed_dict['lengths']  # [batch_size]

        his_vectors = self.i_embeddings(history)
        his_vector = self.encoder(his_vectors, lengths)
        i_vectors = self.i_embeddings(i_ids)
        prediction = (his_vector[:, None, :] * i_vectors).sum(-1)
        out_dict = {'prediction': prediction}

        if self.stage == 1 and feed_dict['phase'] == 'train':
            history_aug = feed_dict['history_items_aug']
            his_aug_vectors = self.i_embeddings(history_aug)
            his_aug_vector = self.encoder(his_aug_vectors, lengths)
            features = torch.stack([his_vector, his_aug_vector], dim=1)  # bsz, 2, emb
            features = F.normalize(features, dim=-1)
            out_dict['features'] = features  # bsz, 2, emb
            out_dict['labels'] = i_ids[:, 0]  # bsz
            # TODO: 根据 future_items 构建稀疏矩阵，并计算 Jaccard 距离

        return out_dict

    def loss(self, out_dict):
        if self.stage == 1:
            loss = self.criterion(out_dict['features'], labels=out_dict['labels'])
        else:
            loss = super().loss(out_dict)
        return loss

    class Dataset(GeneralModel.Dataset):
        @staticmethod
        def reorder_op(seq, ratio):
            select_len = int(len(seq) * ratio)
            start = np.random.randint(0, len(seq) - select_len + 1)
            idx_range = np.arange(len(seq))
            np.random.shuffle(idx_range[start: start + select_len])
            return np.array(seq)[idx_range]

        def _prepare(self):
            # history length must be non-zero
            idx_select = np.array(self.data['position']) > 0
            for key in self.data:
                self.data[key] = np.array(self.data[key])[idx_select]

            # record history sequence
            uid, pos = self.data['user_id'], self.data['position']
            history = list()
            for u, p in zip(uid, pos):
                history_items = np.array([x[0] for x in self.corpus.user_his[u][:p]])
                if self.model.history_max > 0:
                    history_items = history_items[-self.model.history_max:]
                history.append(history_items.tolist())
            self.data['history_items'] = history
            super()._prepare()

        def _get_feed_dict(self, index):
            feed_dict = super()._get_feed_dict(index)
            history_items = np.array(self.data['history_items'][index])
            if self.model.stage in [0, 1] and self.phase == 'train':
                history_items = self.reorder_op(history_items, self.model.reorder_ratio)
                if self.model.stage == 1:
                    history_items_aug = self.reorder_op(history_items, self.model.reorder_ratio)
                    feed_dict['history_items_aug'] = history_items_aug
                    # pos = self.data['position'][index]
                    # user_seq = self.corpus.user_his[feed_dict['user_id']]
                    # future_items = np.array([x[0] for x in user_seq[:-2][pos: pos + self.model.future_window]])
                    # feed_dict['future_items'] = future_items
            feed_dict['history_items'] = history_items
            feed_dict['lengths'] = len(feed_dict['history_items'])
            return feed_dict

        def collate_batch(self, feed_dicts: List[dict]) -> dict:
            # if self.phase == 'train':
            #     intersection = list()
            #     for i in range(len(feed_dicts)):
            #         for j in range(len(feed_dicts)):
            #             if i != j:
            #                 s1, s2 = feed_dicts[i]['future_items'], feed_dicts[j]['future_items']
            #                 inter_size = len(set(s1) & set(s2))
            #                 intersection.append(int(inter_size > 0))
            #     print(np.mean(intersection))
            feed_dict = dict()
            for key in feed_dicts[0]:
                stack_val = np.array([d[key] for d in feed_dicts])
                if stack_val.dtype == np.object:  # inconsistent length (e.g. history)
                    feed_dict[key] = pad_sequence([torch.from_numpy(x) for x in stack_val], batch_first=True)
                else:
                    feed_dict[key] = torch.from_numpy(stack_val)
            feed_dict['batch_size'] = len(feed_dicts)
            feed_dict['phase'] = self.phase
            return feed_dict


""" Soft-Supervised Contrastive Loss"""
class SupConLoss(nn.Module):
    def __init__(self, device, temperature=0.5, contrast_mode='all', base_temperature=1.):
        super(SupConLoss, self).__init__()
        self.device = device
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(self.device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.transpose(0, 1)).float().to(self.device)

        contrast_count = features.shape[1]  # 2
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)  # bsz * 2, -1
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.transpose(0, 1)),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()  # bsz * 2, bsz * 2

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)  # bsz * 2, bsz * 2
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(self.device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-10)

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


""" Encoder Layers """
class GRU4RecEncoder(nn.Module):
    def __init__(self, emb_size, hidden_size):
        super().__init__()
        self.rnn = nn.GRU(input_size=emb_size, hidden_size=hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, emb_size, bias=False)

    def forward(self, seq, lengths):
        # Sort and Pack
        sort_lengths, sort_idx = torch.topk(lengths, k=len(lengths))
        sort_seq = seq.index_select(dim=0, index=sort_idx)
        seq_packed = torch.nn.utils.rnn.pack_padded_sequence(sort_seq, sort_lengths, batch_first=True)

        # RNN
        output, hidden = self.rnn(seq_packed, None)

        # Unsort
        sort_rnn_vector = self.out(hidden[-1])
        unsort_idx = torch.topk(sort_idx, k=len(lengths), largest=False)[1]
        rnn_vector = sort_rnn_vector.index_select(dim=0, index=unsort_idx)

        return rnn_vector


class SASRecEncoder(nn.Module):
    def __init__(self, emb_size, num_layers, num_heads, max_his, dropout, device):
        super().__init__()
        self.device = device

        self.p_embeddings = nn.Embedding(max_his + 1, emb_size)
        self.transformer_block = nn.ModuleList([
            layers.TransformerLayer(d_model=emb_size, d_ff=emb_size, n_heads=num_heads,
                                    dropout=dropout, kq_same=False)
            for _ in range(num_layers)
        ])

    def forward(self, seq, lengths):
        seq_len = seq.size(1)
        len_range = torch.from_numpy(np.arange(seq_len)).to(self.device)
        valid_mask = len_range[None, :] < lengths[:, None]

        # Position embedding
        position = (lengths[:, None] - len_range[None, :]) * valid_mask.long()
        pos_vectors = self.p_embeddings(position)
        seq = seq + pos_vectors

        # Self-attention
        causality_mask = np.tril(np.ones((1, 1, seq_len, seq_len), dtype=np.int))
        attn_mask = torch.from_numpy(causality_mask).to(self.device)
        # attn_mask = valid_his.view(batch_size, 1, 1, seq_len)
        for block in self.transformer_block:
            seq = block(seq, attn_mask)
        seq = seq * valid_mask[:, :, None].float()

        his_vector = (seq * (position == 1).float()[:, :, None]).sum(1)
        return his_vector
