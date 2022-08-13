# -*- coding: UTF-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.BaseModel import SequentialModel
from utils import layers


class CLRec(SequentialModel):
    reader = 'SeqReader'
    runner = 'BaseRunner'
    extra_log_args = ['batch_size', 'temp']

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--emb_size', type=int, default=64,
                            help='Size of embedding vectors.')
        parser.add_argument('--temp', type=float, default=0.2,
                            help='Temperature in contrastive loss.')
        return SequentialModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        super().__init__(args, corpus)
        self.emb_size = args.emb_size
        self.temp = args.temp
        self.max_his = args.history_max
        self._define_params()
        self.apply(self.init_weights)

    def _define_params(self):
        self.i_embeddings = nn.Embedding(self.item_num, self.emb_size)
        self.encoder = BERT4RecEncoder(self.emb_size, self.max_his, num_layers=2, num_heads=2)
        self.contra_loss = ContraLoss(temperature=self.temp)

    def forward(self, feed_dict):
        self.check_list = []
        i_ids = feed_dict['item_id']  # bsz, n_candidate
        history = feed_dict['history_items']  # bsz, history_max
        lengths = feed_dict['lengths']  # bsz

        his_vectors = self.i_embeddings(history)
        his_vector = self.encoder(his_vectors, lengths)
        i_vectors = self.i_embeddings(i_ids)
        # his_vector = F.normalize(his_vector, dim=-1)
        # i_vectors = F.normalize(i_vectors, dim=-1)
        prediction = (his_vector[:, None, :] * i_vectors).sum(-1)
        out_dict = {'prediction': prediction}

        if feed_dict['phase'] == 'train':
            target_vector = i_vectors[:, 0, :]
            features = torch.stack([his_vector, target_vector], dim=1)  # bsz, 2, emb
            features = F.normalize(features, dim=-1)
            out_dict['features'] = features  # bsz, 2, emb

        return out_dict

    def loss(self, out_dict):
        return self.contra_loss(out_dict['features'])

    class Dataset(SequentialModel.Dataset):
        # No need to sample negative items
        def actions_before_epoch(self):
            self.data['neg_items'] = [[] for _ in range(len(self))]


""" Contrastive Loss """
class ContraLoss(nn.Module):
    def __init__(self, temperature=0.2):
        super(ContraLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, mask=None):
        """
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sequence j
                has the same target item as sequence i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size, device = features.shape[0], features.device
        if mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)

        # compute logits
        dot_contrast = torch.matmul(features[:, 0], features[:, 1].transpose(0, 1)) / self.temperature
        # for numerical stability
        logits_max, _ = torch.max(dot_contrast, dim=1, keepdim=True)
        logits = dot_contrast - logits_max.detach()  # bsz, bsz

        # compute log_prob
        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-10)

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1)

        return -mean_log_prob_pos.mean()


""" Encoder Layer """
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
