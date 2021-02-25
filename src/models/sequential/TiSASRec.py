# -*- coding: UTF-8 -*-
# @Author  : Chenyang Wang
# @Email   : THUwangcy@gmail.com

""" TiSASRec
Reference:
    "Time Interval Aware Self-Attention for Sequential Recommendation"
    Jiacheng Li et al., WSDM'2020.
CMD example:
    python main.py --model_name TiSASRec --emb_size 64 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 \
    --history_max 20 --dataset 'Grocery_and_Gourmet_Food'
"""

import torch
import torch.nn as nn
import numpy as np

from models.sequential.SASRec import SASRec


class TiSASRec(SASRec):
    extra_log_args = ['emb_size', 'num_layers', 'num_heads', 'time_max']

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--time_max', type=int, default=512,
                            help='Max time intervals.')
        return SASRec.parse_model_args(parser)

    def __init__(self, args, corpus):
        self.max_time = args.time_max
        super().__init__(args, corpus)

        setattr(corpus, 'user_min_interval', dict())
        for u, user_df in corpus.all_df.groupby('user_id'):
            time_seqs = user_df['time'].values
            interval_matrix = np.abs(time_seqs[:, None] - time_seqs[None, :])
            min_interval = np.min(interval_matrix + (interval_matrix <= 0) * 0xFFFF)
            corpus.user_min_interval[u] = min_interval

    def _define_params(self):
        self.i_embeddings = nn.Embedding(self.item_num, self.emb_size)
        self.p_k_embeddings = nn.Embedding(self.max_his + 1, self.emb_size)
        self.p_v_embeddings = nn.Embedding(self.max_his + 1, self.emb_size)
        self.t_k_embeddings = nn.Embedding(self.max_time + 1, self.emb_size)
        self.t_v_embeddings = nn.Embedding(self.max_time + 1, self.emb_size)

        self.transformer_block = nn.ModuleList([
            TimeIntervalTransformerLayer(d_model=self.emb_size, d_ff=self.emb_size, n_heads=self.num_heads,
                                         dropout=self.dropout, kq_same=False)
            for _ in range(self.num_layers)
        ])

    def forward(self, feed_dict):
        self.check_list = []
        i_ids = feed_dict['item_id']  # [batch_size, -1]
        i_history = feed_dict['history_items']  # [batch_size, history_max]
        t_history = feed_dict['history_times']  # [batch_size, history_max]
        user_min_t = feed_dict['user_min_intervals']  # [batch_size]
        lengths = feed_dict['lengths']  # [batch_size]
        batch_size, seq_len = i_history.shape

        valid_his = (i_history > 0).long()
        his_vectors = self.i_embeddings(i_history)

        # Position embedding
        position = (lengths[:, None] - self.len_range[None, :seq_len]) * valid_his
        pos_k = self.p_k_embeddings(position)
        pos_v = self.p_v_embeddings(position)

        # Interval embedding
        interval_matrix = (t_history[:, :, None] - t_history[:, None, :]).abs()
        interval_matrix = (interval_matrix / user_min_t.view(-1, 1, 1)).long().clamp(0, self.max_time)
        inter_k = self.t_k_embeddings(interval_matrix)
        inter_v = self.t_v_embeddings(interval_matrix)

        # Self-attention
        causality_mask = np.tril(np.ones((1, 1, seq_len, seq_len), dtype=np.int))
        attn_mask = torch.from_numpy(causality_mask).to(self.device)
        # attn_mask = valid_his.view(batch_size, 1, 1, seq_len)
        for block in self.transformer_block:
            his_vectors = block(his_vectors, pos_k, pos_v, inter_k, inter_v, attn_mask)
        his_vectors = his_vectors * valid_his[:, :, None].float()

        his_vector = his_vectors[torch.arange(batch_size), lengths - 1, :]
        # his_vector = his_vectors.sum(1) / lengths[:, None].float()
        # ↑ average pooling is shown to be more effective than the most recent embedding

        i_vectors = self.i_embeddings(i_ids)
        prediction = (his_vector[:, None, :] * i_vectors).sum(-1)
        return {'prediction': prediction.view(batch_size, -1)}

    class Dataset(SASRec.Dataset):
        def _get_feed_dict(self, index):
            feed_dict = super()._get_feed_dict(index)
            user_id = self.data['user_id'][index]
            min_interval = self.corpus.user_min_interval[user_id]
            feed_dict['user_min_intervals'] = min_interval
            return feed_dict


class TimeIntervalMultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, kq_same=False, bias=True):
        super().__init__()
        """
        It also needs position and interaction (time interval) key/value input.
        """
        self.d_model = d_model
        self.h = n_heads
        self.d_k = self.d_model // self.h
        self.kq_same = kq_same

        self.v_linear = nn.Linear(d_model, d_model, bias=bias)
        self.k_linear = nn.Linear(d_model, d_model, bias=bias)
        if not kq_same:
            self.q_linear = nn.Linear(d_model, d_model, bias=bias)

    def forward(self, q, k, v, pos_k, pos_v, inter_k, inter_v, mask):
        bs, seq_len = k.size(0), k.size(1)

        # perform linear operation and split into h heads
        k = (self.k_linear(k) + pos_k).view(bs, seq_len, self.h, self.d_k)
        if not self.kq_same:
            q = self.q_linear(q).view(bs, seq_len, self.h, self.d_k)
        else:
            q = self.k_linear(q).view(bs, seq_len, self.h, self.d_k)
        v = (self.v_linear(v) + pos_v).view(bs, seq_len, self.h, self.d_k)

        # transpose to get dimensions bs * h * -1 * d_k
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        # interaction (time interval) embeddings
        inter_k = inter_k.view(bs, seq_len, seq_len, self.h, self.d_k)
        inter_v = inter_v.view(bs, seq_len, seq_len, self.h, self.d_k)
        inter_k = inter_k.transpose(2, 3).transpose(1, 2)
        inter_v = inter_v.transpose(2, 3).transpose(1, 2)  # bs, head, seq_len, seq_len, d_k

        # calculate attention using function we will define next
        output = self.scaled_dot_product_attention(q, k, v, inter_k, inter_v, self.d_k, mask)

        # concatenate heads and put through final linear layer
        output = output.transpose(1, 2).reshape(bs, -1, self.d_model)
        return output

    @staticmethod
    def scaled_dot_product_attention(q, k, v, inter_k, inter_v, d_k, mask):
        """
        Involve pair interaction embeddings when calculating attention scores and output
        """
        scores = torch.matmul(q, k.transpose(-2, -1))  # bs, head, q_len, k_len
        scores += (q[:, :, :, None, :] * inter_k).sum(-1)
        scores = scores / d_k ** 0.5
        scores.masked_fill_(mask == 0, -np.inf)
        scores = (scores - scores.max()).softmax(dim=-1)
        output = torch.matmul(scores, v)  # bs, head, q_len, d_k
        output += (scores[:, :, :, :, None] * inter_v).sum(-2)
        return output


class TimeIntervalTransformerLayer(nn.Module):
    def __init__(self, d_model, d_ff, n_heads, dropout, kq_same=False):
        super().__init__()
        self.masked_attn_head = TimeIntervalMultiHeadAttention(d_model, n_heads, kq_same=kq_same)

        # Two layer norm layer and two dropout layer
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, seq, pos_k, pos_v, inter_k, inter_v, mask):
        context = self.masked_attn_head(seq, seq, seq, pos_k, pos_v, inter_k, inter_v, mask)
        context = self.layer_norm1(self.dropout1(context) + seq)
        output = self.linear1(context).relu()
        output = self.linear2(output)
        output = self.layer_norm2(self.dropout2(output) + context)
        return output
