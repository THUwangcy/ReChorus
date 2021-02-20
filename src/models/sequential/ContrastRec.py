# -*- coding: UTF-8 -*-

import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.BaseModel import SequentialModel
from utils import layers
from utils import utils


class ContrastRec(SequentialModel):
    extra_log_args = ['stage', 'gamma', 'batch_size', 'temp', 'encoder']

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--stage', type=int, default=0, choices=[0, 1, 2],
                            help='Stage of training: 0-joint, 1-contrastive, 2-recommendation.')
        parser.add_argument('--emb_size', type=int, default=64,
                            help='Size of embedding vectors.')
        parser.add_argument('--gamma', type=float, default=1,
                            help='Coefficient of the contrastive loss.')
        parser.add_argument('--beta_a', type=int, default=3,
                            help='Parameter of the beta distribution for sampling.')
        parser.add_argument('--beta_b', type=int, default=3,
                            help='Parameter of the beta distribution for sampling.')
        parser.add_argument('--temp', type=float, default=0.2,
                            help='Temperature in contrastive loss.')
        parser.add_argument('--encoder', type=str, default='BERT4Rec',
                            help='Choose a sequence encoder: GRU4Rec, Caser, BERT4Rec.')
        parser.add_argument('--checkpoint', type=str, default='',
                            help='Choose a pre-train model checkpoint.')
        return SequentialModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        self.stage = args.stage
        self.emb_size = args.emb_size
        self.max_his = args.history_max
        self.gamma = args.gamma
        self.beta_a = args.beta_a
        self.beta_b = args.beta_b
        self.temperature = args.temp
        self.encoder_name = args.encoder
        self.pre_path = args.checkpoint
        if self.pre_path == '':
            self.pre_path = '../model/ContrastRec/Pre__{}__{}__encoder={}__temp={}__bsz={}.pt'.format(
                corpus.dataset, args.random_seed, self.encoder_name, self.temperature, args.batch_size)
        super().__init__(args, corpus)

    def actions_before_train(self):
        if self.stage == 1:
            self.model_path = self.pre_path
        elif self.stage == 2:
            if os.path.exists(self.pre_path):
                self.load_model(self.pre_path)
            else:
                msg = 'Train from scratch because pre-train model does not exist: '
                logging.warning(msg + self.pre_path)

    def _define_params(self):
        self.i_embeddings = nn.Embedding(self.item_num, self.emb_size)
        if self.encoder_name == 'GRU4Rec':
            self.encoder = GRU4RecEncoder(self.emb_size, hidden_size=128)
        elif self.encoder_name == 'Caser':
            self.encoder = CaserEncoder(self.emb_size, self.max_his, num_horizon=16, num_vertical=8, l=5)
        elif self.encoder_name == 'BERT4Rec':
            self.encoder = BERT4RecEncoder(self.emb_size, self.max_his, self.device, num_layers=2, num_heads=2)
        else:
            raise ValueError('Invalid sequence encoder.')
        self.criterion = SupConLoss(self.device, temperature=self.temperature)
        # self.head = nn.Linear(self.emb_size, self.emb_size)

    def forward(self, feed_dict):
        self.check_list = []
        i_ids = feed_dict['item_id']  # bsz, n_candidate
        history = feed_dict['history_items']  # bsz, history_max
        lengths = feed_dict['lengths']  # bsz

        his_vectors = self.i_embeddings(history)
        his_vector = self.encoder(his_vectors, lengths)
        i_vectors = self.i_embeddings(i_ids)
        prediction = (his_vector[:, None, :] * i_vectors).sum(-1)
        out_dict = {'prediction': prediction}

        if self.stage in [0, 1] and feed_dict['phase'] == 'train':
            history_aug = feed_dict['history_items_aug']
            his_aug_vectors = self.i_embeddings(history_aug)
            his_aug_vector = self.encoder(his_aug_vectors, lengths)
            features = torch.stack([his_vector, his_aug_vector], dim=1)  # bsz, 2, emb
            # features = self.head(features)
            features = F.normalize(features, dim=-1)
            out_dict['features'] = features  # bsz, 2, emb
            out_dict['labels'] = i_ids[:, 0]  # bsz

        return out_dict

    def loss(self, out_dict):
        if self.stage == 0:
            contrastive_loss = self.criterion(out_dict['features'], labels=out_dict['labels'])
            loss = super().loss(out_dict) + self.gamma * contrastive_loss
        elif self.stage == 1:
            loss = self.criterion(out_dict['features'], labels=out_dict['labels'])
        else:
            loss = super().loss(out_dict)
        return loss

    class Dataset(SequentialModel.Dataset):
        # def _prepare(self):
        #     if self.phase == 'train':
        #         self.data = utils.df_to_dict(self.corpus.data_df['train'].sample(frac=0.8))
        #     super()._prepare()

        def reorder_op(self, seq):
            ratio = np.random.beta(a=self.model.beta_a, b=self.model.beta_b)
            select_len = int(len(seq) * ratio)
            start = np.random.randint(0, len(seq) - select_len + 1)
            idx_range = np.arange(len(seq))
            np.random.shuffle(idx_range[start: start + select_len])
            return np.array(seq)[idx_range]

        def _get_feed_dict(self, index):
            feed_dict = super()._get_feed_dict(index)
            if self.model.stage in [0, 1] and self.phase == 'train':
                history_items = self.reorder_op(feed_dict['history_items'])
                history_items_aug = self.reorder_op(feed_dict['history_items'])
                feed_dict['history_items'] = history_items
                feed_dict['history_items_aug'] = history_items_aug
            return feed_dict


""" Supervised Contrastive Loss """
class SupConLoss(nn.Module):
    def __init__(self, device, temperature=0.5, contrast_mode='all'):
        super(SupConLoss, self).__init__()
        self.device = device
        self.temperature = temperature
        self.contrast_mode = contrast_mode

    def forward(self, features, labels=None, mask=None):
        """
        Compute loss for model. If both `labels` and `mask` are None,
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
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-10)

        # loss
        loss = - self.temperature * mean_log_prob_pos
        return loss.mean()

""" Encoder Layers """
class GRU4RecEncoder(nn.Module):
    def __init__(self, emb_size, hidden_size=128):
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

class CaserEncoder(nn.Module):
    def __init__(self, emb_size, max_his, num_horizon=16, num_vertical=8, l=5):
        super().__init__()
        self.max_his = max_his
        lengths = [i + 1 for i in range(l)]
        self.conv_h = nn.ModuleList(
            [nn.Conv2d(in_channels=1, out_channels=num_horizon, kernel_size=(i, emb_size)) for i in lengths])
        self.conv_v = nn.Conv2d(in_channels=1, out_channels=num_vertical, kernel_size=(max_his, 1))
        self.fc_dim_h = num_horizon * len(lengths)
        self.fc_dim_v = num_vertical * emb_size
        fc_dim_in = self.fc_dim_v + self.fc_dim_h
        self.fc = nn.Linear(fc_dim_in, emb_size)

    def forward(self, seq, lengths):
        batch_size, seq_len = seq.size(0), seq.size(1)
        pad_len = self.max_his - seq_len
        seq = F.pad(seq, [0, 0, 0, pad_len]).unsqueeze(1)

        # Convolution Layers
        out_v = self.conv_v(seq).view(-1, self.fc_dim_v)
        out_hs = list()
        for conv in self.conv_h:
            conv_out = conv(seq).squeeze(3).relu()
            pool_out = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
            out_hs.append(pool_out)
        out_h = torch.cat(out_hs, 1)

        # Fully-connected Layers
        his_vector = self.fc(torch.cat([out_v, out_h], 1))
        return his_vector

class BERT4RecEncoder(nn.Module):
    def __init__(self, emb_size, max_his, device, num_layers=2, num_heads=2):
        super().__init__()
        self.device = device

        self.p_embeddings = nn.Embedding(max_his + 1, emb_size)
        self.transformer_block = nn.ModuleList([
            layers.TransformerLayer(d_model=emb_size, d_ff=emb_size, n_heads=num_heads)
            for _ in range(num_layers)
        ])

    def forward(self, seq, lengths):
        batch_size, seq_len = seq.size(0), seq.size(1)
        len_range = torch.from_numpy(np.arange(seq_len)).to(self.device)
        valid_mask = len_range[None, :] < lengths[:, None]

        # Position embedding
        position = len_range[None, :] * valid_mask.long()
        pos_vectors = self.p_embeddings(position)
        seq = seq + pos_vectors

        # Self-attention
        attn_mask = valid_mask.view(batch_size, 1, 1, seq_len)
        for block in self.transformer_block:
            seq = block(seq, attn_mask)
        seq = seq * valid_mask[:, :, None].float()

        his_vector = seq[torch.arange(batch_size), lengths - 1]
        return his_vector
