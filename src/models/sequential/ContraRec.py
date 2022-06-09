# -*- coding: UTF-8 -*-
# @Author  : Chenyang Wang
# @Email   : THUwangcy@gmail.com

""" ContraRec
Reference:
    "Sequential Recommendation with Multiple Contrast Signals"
    Wang et al., TOIS'2022.
CMD example:
    python main.py --model_name ContraRec --emb_size 64 --lr 1e-4 --l2 1e-6 --history_max 20 --encoder BERT4Rec \
    --num_neg 1 --ctc_temp 1 --ccc_temp 0.2 --batch_size 4096 --gamma 1 --dataset Grocery_and_Gourmet_Food
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.BaseModel import SequentialModel
from utils import layers


class ContraRec(SequentialModel):
    reader = 'SeqReader'
    runner = 'BaseRunner'
    extra_log_args = ['gamma', 'num_neg', 'batch_size', 'ctc_temp', 'ccc_temp', 'encoder']

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--emb_size', type=int, default=64,
                            help='Size of embedding vectors.')
        parser.add_argument('--gamma', type=float, default=1,
                            help='Coefficient of the contrastive loss.')
        parser.add_argument('--beta_a', type=int, default=3,
                            help='Parameter of the beta distribution for sampling.')
        parser.add_argument('--beta_b', type=int, default=3,
                            help='Parameter of the beta distribution for sampling.')
        parser.add_argument('--ctc_temp', type=float, default=1,
                            help='Temperature in context-target contrastive loss.')
        parser.add_argument('--ccc_temp', type=float, default=0.2,
                            help='Temperature in context-context contrastive loss.')
        parser.add_argument('--encoder', type=str, default='BERT4Rec',
                            help='Choose a sequence encoder: GRU4Rec, Caser, BERT4Rec.')
        return SequentialModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        super().__init__(args, corpus)
        self.emb_size = args.emb_size
        self.max_his = args.history_max
        self.gamma = args.gamma
        self.beta_a = args.beta_a
        self.beta_b = args.beta_b
        self.ctc_temp = args.ctc_temp
        self.ccc_temp = args.ccc_temp
        self.encoder_name = args.encoder
        self.mask_token = corpus.n_items
        self._define_params()
        self.apply(self.init_weights)

    def _define_params(self):
        self.i_embeddings = nn.Embedding(self.item_num + 1, self.emb_size)
        if self.encoder_name == 'GRU4Rec':
            self.encoder = GRU4RecEncoder(self.emb_size, hidden_size=128)
        elif self.encoder_name == 'Caser':
            self.encoder = CaserEncoder(self.emb_size, self.max_his, num_horizon=16, num_vertical=8, l=5)
        elif self.encoder_name == 'BERT4Rec':
            self.encoder = BERT4RecEncoder(self.emb_size, self.max_his, num_layers=2, num_heads=2)
        else:
            raise ValueError('Invalid sequence encoder.')
        self.ccc_loss = ContraLoss(self.device, temperature=self.ccc_temp)

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

        if feed_dict['phase'] == 'train':
            history_a = feed_dict['history_items_a']
            his_a_vectors = self.i_embeddings(history_a)
            his_a_vector = self.encoder(his_a_vectors, lengths)
            history_b = feed_dict['history_items_b']
            his_b_vectors = self.i_embeddings(history_b)
            his_b_vector = self.encoder(his_b_vectors, lengths)
            features = torch.stack([his_a_vector, his_b_vector], dim=1)  # bsz, 2, emb
            features = F.normalize(features, dim=-1)
            out_dict['features'] = features  # bsz, 2, emb
            out_dict['labels'] = i_ids[:, 0]  # bsz

        return out_dict

    def loss(self, out_dict):
        predictions = out_dict['prediction'] / self.ctc_temp
        pre_softmax = (predictions - predictions.max()).softmax(dim=1)
        ctc_loss = - self.ctc_temp * pre_softmax[:, 0].log().mean()
        ccc_loss = self.ccc_loss(out_dict['features'], labels=out_dict['labels'])
        loss = ctc_loss + self.gamma * ccc_loss
        return loss

    class Dataset(SequentialModel.Dataset):
        def reorder_op(self, seq):
            ratio = np.random.beta(a=self.model.beta_a, b=self.model.beta_b)
            select_len = int(len(seq) * ratio)
            start = np.random.randint(0, len(seq) - select_len + 1)
            idx_range = np.arange(len(seq))
            np.random.shuffle(idx_range[start: start + select_len])
            return seq[idx_range]

        def mask_op(self, seq):
            ratio = np.random.beta(a=self.model.beta_a, b=self.model.beta_b)
            selected_len = int(len(seq) * ratio)
            mask = np.full(len(seq), False)
            mask[:selected_len] = True
            np.random.shuffle(mask)
            seq[mask] = self.model.mask_token
            return seq

        def augment(self, seq):
            aug_seq = np.array(seq).copy()
            if np.random.rand() > 0.5:
                return self.mask_op(aug_seq)
            else:
                return self.reorder_op(aug_seq)

        def _get_feed_dict(self, index):
            feed_dict = super()._get_feed_dict(index)
            if self.phase == 'train':
                history_items_a = self.augment(feed_dict['history_items'])
                history_items_b = self.augment(feed_dict['history_items'])
                feed_dict['history_items_a'] = history_items_a
                feed_dict['history_items_b'] = history_items_b
            return feed_dict


""" Context-Context Contrastive Loss """
class ContraLoss(nn.Module):
    def __init__(self, device, temperature=0.2):
        super(ContraLoss, self).__init__()
        self.device = device
        self.temperature = temperature

    def forward(self, features, labels=None):
        """
        If both `labels` and `mask` are None, it degenerates to InfoNCE loss
        Args:
            features: hidden vector of shape [bsz, n_views, dim].
            labels: target item of shape [bsz].
        Returns:
            A loss scalar.
        """
        batch_size = features.shape[0]
        if labels is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(self.device)
        else:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.transpose(0, 1)).float().to(self.device)

        contrast_count = features.shape[1]  # n_views
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)  # bsz * n_views, -1

        # compute logits
        anchor_dot_contrast = torch.matmul(contrast_feature, contrast_feature.transpose(0, 1)) /self.temperature
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()  # bsz * n_views, bsz * n_views

        # tile mask
        mask = mask.repeat(contrast_count, contrast_count)  # bsz * n_views, bsz * n_views
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask), 1,
            torch.arange(mask.shape[0]).view(-1, 1).to(self.device), 0
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
        seq_packed = torch.nn.utils.rnn.pack_padded_sequence(sort_seq, sort_lengths.cpu(), batch_first=True)

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
    def __init__(self, emb_size, max_his, num_layers=2, num_heads=2):
        super().__init__()
        self.p_embeddings = nn.Embedding(max_his + 1, emb_size)
        self.transformer_block = nn.ModuleList([
            layers.TransformerLayer(d_model=emb_size, d_ff=emb_size, n_heads=num_heads)
            for _ in range(num_layers)
        ])

    def forward(self, seq, lengths):
        batch_size, seq_len = seq.size(0), seq.size(1)
        len_range = torch.from_numpy(np.arange(seq_len)).to(seq.device)
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
