# -*- coding: UTF-8 -*-

import torch
import numpy as np

from models.SASRec import SASRec


class TiSASRec(SASRec):
    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--time_max', type=int, default=256,
                            help='Max time intervals.')
        return SASRec.parse_model_args(parser)

    def __init__(self, args, corpus):
        self.max_time = args.time_max

        setattr(corpus, 'user_min_interval', dict())
        for u, user_df in corpus.all_df.groupby('user_id'):
            time_seqs = user_df['time'].values
            interval_matrix = np.abs(time_seqs[:, None] - time_seqs[None, :])
            min_interval = np.min(interval_matrix + (interval_matrix <= 0) * 0xFFFF)
            corpus.user_min_interval[u] = min_interval

        super().__init__(args, corpus)

    def _define_params(self):
        self.i_embeddings = torch.nn.Embedding(self.item_num, self.emb_size)
        self.p_k_embeddings = torch.nn.Embedding(self.max_his + 1, self.emb_size)
        self.p_v_embeddings = torch.nn.Embedding(self.max_his + 1, self.emb_size)
        self.t_k_embeddings = torch.nn.Embedding(self.max_time + 1, self.emb_size)
        self.t_v_embeddings = torch.nn.Embedding(self.max_time + 1, self.emb_size)

        self.Q = torch.nn.Linear(self.emb_size, self.emb_size, bias=False)
        self.K = torch.nn.Linear(self.emb_size, self.emb_size, bias=False)
        self.V = torch.nn.Linear(self.emb_size, self.emb_size, bias=False)
        self.W1 = torch.nn.Linear(self.emb_size, self.emb_size)
        self.W2 = torch.nn.Linear(self.emb_size, self.emb_size)

        self.dropout_layer = torch.nn.Dropout(p=self.dropout)
        self.layer_norm = torch.nn.LayerNorm(self.emb_size)

    def forward(self, feed_dict):
        self.check_list = []
        i_ids = feed_dict['item_id']                  # [batch_size, -1]
        i_history = feed_dict['history_items']        # [batch_size, history_max]
        t_history = feed_dict['history_times']        # [batch_size, history_max]
        user_min_t = feed_dict['user_min_intervals']  # [batch_size]
        lengths = feed_dict['lengths']                # [batch_size]
        batch_size, seq_len = i_history.shape

        valid_his = (i_history > 0).byte()
        i_vectors = self.i_embeddings(i_ids)
        his_vectors = self.i_embeddings(i_history)

        # Position embedding
        # lengths:  [4, 2, 5]
        # position: [[4, 3, 2, 1, 0], [2, 1, 0, 0, 0], [5, 4, 3, 2, 1]]
        position = self.len_range[:seq_len].unsqueeze(0).repeat(batch_size, 1)
        position = (lengths[:, None] - position) * valid_his.long()
        pos_k_vectors = self.p_k_embeddings(position)
        pos_v_vectors = self.p_v_embeddings(position)

        # Interval embedding
        interval_matrix = (t_history[:, :, None] - t_history[:, None, :]).abs()
        interval_matrix = (interval_matrix / user_min_t.view(-1, 1, 1)).long().clamp(0, self.max_time)
        inter_k_vectors = self.t_k_embeddings(interval_matrix)
        inter_v_vectors = self.t_v_embeddings(interval_matrix)

        # Self-attention
        attn_mask = 1 - valid_his.unsqueeze(1).repeat(1, seq_len, 1)
        for i in range(self.num_layers):
            residual = his_vectors
            query = self.Q(his_vectors)
            key = self.K(his_vectors) + pos_k_vectors
            value = self.V(his_vectors) + pos_v_vectors  # [batch_size, history_max, emb_size]
            attention = torch.bmm(query, key.transpose(1, 2))  # [batch_size, history_max, history_max]
            attention += (query[:, :, None, :] * inter_k_vectors).sum(-1)
            attention = attention * (self.emb_size ** -0.5)
            attention = (attention - attention.max()).masked_fill(attn_mask, -np.inf)
            attention = attention.softmax(dim=-1)
            context = torch.bmm(attention, value)  # [batch_size, history_max, emb_size]
            context += (attention[:, :, :, None] * inter_v_vectors).sum(2)
            # mlp forward
            his_vectors = self.W1(context).relu()
            his_vectors = self.W2(his_vectors)
            his_vectors = self.dropout_layer(his_vectors)
            his_vectors = self.layer_norm(residual + his_vectors)
            # ↑ layer norm in the end is shown to be more effective
        his_vectors = his_vectors * valid_his[:, :, None].double()

        # his_vector = (his_vectors * (position == 1).double()[:, :, None]).sum(1)
        his_vector = his_vectors.sum(1) / lengths[:, None].double()
        # ↑ average pooling is shown to be more effective than the most recent embedding

        prediction = (his_vector[:, None, :] * i_vectors).sum(-1)
        return prediction.view(batch_size, -1)

    class Dataset(SASRec.Dataset):
        def _get_feed_dict(self, index):
            feed_dict = super()._get_feed_dict(index)
            user_id = self.data['user_id'][index]
            min_interval = self.corpus.user_min_interval[user_id]
            feed_dict['history_times'] = np.array(self.data['time_his'][index])
            feed_dict['user_min_intervals'] = min_interval
            return feed_dict
