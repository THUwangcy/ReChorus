# -*- coding: UTF-8 -*-

import torch
import numpy as np

from models.GRU4Rec import GRU4Rec
from utils import components
from utils import utils


class SASRec(GRU4Rec):
    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--num_layers', type=int, default=1,
                            help='Number of self-attention layers.')
        return GRU4Rec.parse_model_args(parser)

    def __init__(self, args, corpus):
        self.max_his = args.history_max
        self.num_layers = args.num_layers
        super().__init__(args, corpus)
        self.len_range = torch.from_numpy(np.arange(self.max_his)).to(self.device)

    def _define_params(self):
        self.i_embeddings = torch.nn.Embedding(self.item_num, self.emb_size)
        self.p_embeddings = torch.nn.Embedding(self.max_his + 1, self.emb_size)

        self.Q = torch.nn.Linear(self.emb_size, self.emb_size, bias=False)
        self.K = torch.nn.Linear(self.emb_size, self.emb_size, bias=False)
        self.V = torch.nn.Linear(self.emb_size, self.emb_size, bias=False)
        self.W1 = torch.nn.Linear(self.emb_size, self.emb_size)
        self.W2 = torch.nn.Linear(self.emb_size, self.emb_size)

        self.dropout_layer = torch.nn.Dropout(p=self.dropout)
        self.layer_norm = torch.nn.LayerNorm(self.emb_size)

    def forward(self, feed_dict):
        self.check_list = []
        i_ids = feed_dict['item_id']          # [batch_size, -1]
        history = feed_dict['history_items']  # [batch_size, history_max]
        lengths = feed_dict['lengths']        # [batch_size]
        batch_size, seq_len = history.shape

        valid_his = (history > 0).byte()
        i_vectors = self.i_embeddings(i_ids)
        his_vectors = self.i_embeddings(history)

        # Position embedding
        # lengths:  [4, 2, 5]
        # position: [[4, 3, 2, 1, 0], [2, 1, 0, 0, 0], [5, 4, 3, 2, 1]]
        position = self.len_range[:seq_len].unsqueeze(0).repeat(batch_size, 1)
        position = (lengths[:, None] - position) * valid_his.long()
        pos_vectors = self.p_embeddings(position)
        his_vectors = his_vectors + pos_vectors

        # Self-attention
        attn_mask = 1 - valid_his.unsqueeze(1).repeat(1, seq_len, 1)
        for i in range(self.num_layers):
            residual = his_vectors
            # self-attention
            query, key, value = self.Q(his_vectors), self.K(his_vectors), self.V(his_vectors)
            scale = self.emb_size ** -0.5
            his_vectors = components.scaled_dot_product_attention(
                query, key, value, scale=scale, attn_mask=attn_mask)
            # mlp forward
            his_vectors = self.W1(his_vectors).relu()
            his_vectors = self.W2(his_vectors)  # [batch_size, history_max, emb_size]
            # dropout, residual and layer_norm
            his_vectors = self.dropout_layer(his_vectors)
            his_vectors = self.layer_norm(residual + his_vectors)
            # ↑ layer norm in the end is shown to be more effective
        his_vectors = his_vectors * valid_his[:, :, None].float()

        # his_vector = (his_vectors * (position == 1).float()[:, :, None]).sum(1)
        his_vector = his_vectors.sum(1) / lengths[:, None].float()
        # ↑ average pooling is shown to be more effective than the most recent embedding

        prediction = (his_vector[:, None, :] * i_vectors).sum(-1)
        return prediction.view(batch_size, -1)
