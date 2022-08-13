# -*- coding: UTF-8 -*-
# @Author  : Chenyang Wang
# @Email   : THUwangcy@gmail.com

""" KDA
Reference:
    "Toward Dynamic User Intention: Temporal Evolutionary Effects of Item Relations in Sequential Recommendation"
    Chenyang Wang et al., TOIS'2021.
CMD example:
    python main.py --model_name KDA --emb_size 64 --include_attr 1 --freq_rand 0 --lr 1e-3 --l2 1e-6 --num_heads 4 \
    --history_max 20 --dataset 'Grocery_and_Gourmet_Food'
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from utils import layers
from models.BaseModel import SequentialModel
from helpers.KDAReader import KDAReader


class KDA(SequentialModel):
    reader = 'KDAReader'
    runner = 'BaseRunner'
    extra_log_args = ['num_layers', 'num_heads', 'gamma', 'freq_rand', 'include_val']

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--emb_size', type=int, default=64,
                            help='Size of embedding vectors.')
        parser.add_argument('--neg_head_p', type=float, default=0.5,
                            help='The probability of sampling negative head entity.')
        parser.add_argument('--num_layers', type=int, default=1,
                            help='Number of self-attention layers.')
        parser.add_argument('--num_heads', type=int, default=1,
                            help='Number of attention heads.')
        parser.add_argument('--gamma', type=float, default=-1,
                            help='Coefficient of KG loss (-1 for auto-determine).')
        parser.add_argument('--attention_size', type=int, default=10,
                            help='Size of attention hidden space.')
        parser.add_argument('--pooling', type=str, default='average',
                            help='Method of pooling relational history embeddings: average, max, attention')
        parser.add_argument('--include_val', type=int, default=1,
                            help='Whether include relation value in the relation representation')
        return SequentialModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        super().__init__(args, corpus)
        self.relation_num = corpus.n_relations
        self.entity_num = corpus.n_entities
        self.freq_x = corpus.freq_x
        self.freq_dim = args.n_dft // 2 + 1
        self.freq_rand = args.freq_rand
        self.emb_size = args.emb_size
        self.neg_head_p = args.neg_head_p
        self.layer_num = args.num_layers
        self.head_num = args.num_heads
        self.attention_size = args.attention_size
        self.pooling = args.pooling.lower()
        self.include_val = args.include_val
        self.gamma = args.gamma
        if self.gamma < 0:
            self.gamma = len(corpus.relation_df) / len(corpus.all_df)
        self._define_params()
        self.apply(self.init_weights)

        if not self.freq_rand:
            dft_freq_real = torch.tensor(np.real(self.freq_x))  # R * n_freq
            dft_freq_imag = torch.tensor(np.imag(self.freq_x))
            self.relational_dynamic_aggregation.freq_real.weight.data.copy_(dft_freq_real)
            self.relational_dynamic_aggregation.freq_imag.weight.data.copy_(dft_freq_imag)

    def _define_params(self):
        self.user_embeddings = nn.Embedding(self.user_num, self.emb_size)
        self.entity_embeddings = nn.Embedding(self.entity_num, self.emb_size)
        self.relation_embeddings = nn.Embedding(self.relation_num, self.emb_size)
        # First-level aggregation
        self.relational_dynamic_aggregation = RelationalDynamicAggregation(
            self.relation_num, self.freq_dim, self.relation_embeddings, self.include_val, self.device
        )
        # Second-level aggregation
        self.attn_head = layers.MultiHeadAttention(self.emb_size, self.head_num, bias=False)
        self.W1 = nn.Linear(self.emb_size, self.emb_size)
        self.W2 = nn.Linear(self.emb_size, self.emb_size)
        self.dropout_layer = nn.Dropout(self.dropout)
        self.layer_norm = nn.LayerNorm(self.emb_size)
        # Pooling
        if self.pooling == 'attention':
            self.A = nn.Linear(self.emb_size, self.attention_size)
            self.A_out = nn.Linear(self.attention_size, 1, bias=False)
        # Prediction
        self.item_bias = nn.Embedding(self.item_num, 1)

    def forward(self, feed_dict):
        self.check_list = []
        prediction = self.rec_forward(feed_dict)
        out_dict = {'prediction': prediction}
        if feed_dict['phase'] == 'train':
            kg_prediction = self.kg_forward(feed_dict)
            out_dict['kg_prediction'] = kg_prediction
        return out_dict

    def rec_forward(self, feed_dict):
        u_ids = feed_dict['user_id']  # B
        i_ids = feed_dict['item_id']  # B * -1
        v_ids = feed_dict['item_val']  # B * -1 * R
        history = feed_dict['history_items']  # B * H
        delta_t_n = feed_dict['history_delta_t'].float()  # B * H
        batch_size, seq_len = history.shape

        u_vectors = self.user_embeddings(u_ids)
        i_vectors = self.entity_embeddings(i_ids)
        v_vectors = self.entity_embeddings(v_ids)  # B * -1 * R * V
        his_vectors = self.entity_embeddings(history)  # B * H * V

        """
        Relational Dynamic History Aggregation
        """
        valid_mask = (history > 0).view(batch_size, 1, seq_len, 1)
        context = self.relational_dynamic_aggregation(
            his_vectors, delta_t_n, i_vectors, v_vectors, valid_mask)  # B * -1 * R * V

        """
        Multi-layer Self-attention
        """
        for i in range(self.layer_num):
            residual = context
            # self-attention
            context = self.attn_head(context, context, context)
            # feed forward
            context = self.W1(context)
            context = self.W2(context.relu())
            # dropout, residual and layer_norm
            context = self.dropout_layer(context)
            context = self.layer_norm(residual + context)

        """
        Pooling Layer
        """
        if self.pooling == 'attention':
            query_vectors = context * u_vectors[:, None, None, :]  # B * -1 * R * V
            user_attention = self.A_out(self.A(query_vectors).tanh()).squeeze(-1)  # B * -1 * R
            user_attention = (user_attention - user_attention.max()).softmax(dim=-1)
            his_vector = (context * user_attention[:, :, :, None]).sum(dim=-2)  # B * -1 * V
        elif self.pooling == 'max':
            his_vector = context.max(dim=-2).values  # B * -1 * V
        else:
            his_vector = context.mean(dim=-2)  # B * -1 * V

        """
        Prediction
        """
        i_bias = self.item_bias(i_ids).squeeze(-1)
        prediction = ((u_vectors[:, None, :] + his_vector) * i_vectors).sum(dim=-1)
        prediction = prediction + i_bias
        return prediction.view(feed_dict['batch_size'], -1)

    def kg_forward(self, feed_dict):
        head_ids = feed_dict['head_id'].long()  # B * -1
        tail_ids = feed_dict['tail_id'].long()  # B * -1
        value_ids = feed_dict['value_id'].long()  # B
        relation_ids = feed_dict['relation_id'].long()  # B

        head_vectors = self.entity_embeddings(head_ids)
        tail_vectors = self.entity_embeddings(tail_ids)
        value_vectors = self.entity_embeddings(value_ids)
        relation_vectors = self.relation_embeddings(relation_ids)

        # DistMult
        if self.include_val:
            prediction = (head_vectors * (relation_vectors + value_vectors)[:, None, :] * tail_vectors).sum(-1)
        else:
            prediction = (head_vectors * relation_vectors[:, None, :] * tail_vectors).sum(-1)
        return prediction

    def loss(self, out_dict):
        predictions = out_dict['prediction']
        pos_pred, neg_pred = predictions[:, 0], predictions[:, 1:]
        neg_softmax = (neg_pred - neg_pred.max()).softmax(dim=1)
        rec_loss = -((pos_pred[:, None] - neg_pred).sigmoid() * neg_softmax).sum(dim=1).log().mean()

        predictions = out_dict['kg_prediction']
        pos_pred, neg_pred = predictions[:, 0], predictions[:, 1:]
        neg_softmax = (neg_pred - neg_pred.max()).softmax(dim=1)
        kg_loss = -((pos_pred[:, None] - neg_pred).sigmoid() * neg_softmax).sum(dim=1).log().mean()

        loss = rec_loss + self.gamma * kg_loss
        return loss

    class Dataset(SequentialModel.Dataset):
        def __init__(self, model, corpus, phase):
            super().__init__(model, corpus, phase)
            if self.phase == 'train':
                self.kg_data, self.neg_heads, self.neg_tails = None, None, None

            # Prepare item-to-value dict
            item_val = self.corpus.item_meta_df.copy()
            item_val[self.corpus.item_relations] = 0  # set the value of natural item relations to None
            for idx, r in enumerate(self.corpus.attr_relations):
                base = self.corpus.n_items + np.sum(self.corpus.attr_max[:idx])
                item_val[r] = item_val[r].apply(lambda x: x + base).astype(int)
            item_vals = item_val[self.corpus.relations].values  # this ensures the order is consistent to relations
            self.item_val_dict = dict()
            for item, vals in zip(item_val['item_id'].values, item_vals.tolist()):
                self.item_val_dict[item] = [0] + vals  # the first dimension None for the virtual relation

        def _get_feed_dict(self, index):
            feed_dict = super()._get_feed_dict(index)
            feed_dict['item_val'] = [self.item_val_dict[item] for item in feed_dict['item_id']]
            delta_t = self.data['time'][index] - feed_dict['history_times']
            feed_dict['history_delta_t'] = KDAReader.norm_time(delta_t, self.corpus.t_scalar)
            if self.phase == 'train':
                feed_dict['head_id'] = np.concatenate([[self.kg_data['head'][index]], self.neg_heads[index]])
                feed_dict['tail_id'] = np.concatenate([[self.kg_data['tail'][index]], self.neg_tails[index]])
                feed_dict['relation_id'] = self.kg_data['relation'][index]
                feed_dict['value_id'] = self.kg_data['value'][index]
            return feed_dict

        def generate_kg_data(self) -> pd.DataFrame:
            rec_data_size = len(self)
            replace = (rec_data_size > len(self.corpus.relation_df))
            kg_data = self.corpus.relation_df.sample(n=rec_data_size, replace=replace).reset_index(drop=True)
            kg_data['value'] = np.zeros(len(kg_data), dtype=int)  # default for None
            tail_select = kg_data['tail'].apply(lambda x: x < self.corpus.n_items)
            item_item_df = kg_data[tail_select]
            item_attr_df = kg_data.drop(item_item_df.index)
            item_attr_df['value'] = item_attr_df['tail'].values

            sample_tails = list()  # sample items sharing the same attribute
            for head, val in zip(item_attr_df['head'].values, item_attr_df['tail'].values):
                share_attr_items = self.corpus.share_attr_dict[val]
                tail_idx = np.random.randint(len(share_attr_items))
                sample_tails.append(share_attr_items[tail_idx])
            item_attr_df['tail'] = sample_tails
            kg_data = pd.concat([item_item_df, item_attr_df], ignore_index=True)
            return kg_data

        def actions_before_epoch(self):
            super().actions_before_epoch()
            self.kg_data = self.generate_kg_data()
            heads, tails = self.kg_data['head'].values, self.kg_data['tail'].values
            relations, vals = self.kg_data['relation'].values, self.kg_data['value'].values
            self.neg_heads = np.random.randint(1, self.corpus.n_items, size=(len(self.kg_data), self.model.num_neg))
            self.neg_tails = np.random.randint(1, self.corpus.n_items, size=(len(self.kg_data), self.model.num_neg))
            for i in range(len(self.kg_data)):
                item_item_relation = (tails[i] <= self.corpus.n_items)
                for j in range(self.model.num_neg):
                    if np.random.rand() < self.model.neg_head_p:  # sample negative head
                        tail = tails[i] if item_item_relation else vals[i]
                        while (self.neg_heads[i][j], relations[i], tail) in self.corpus.triplet_set:
                            self.neg_heads[i][j] = np.random.randint(1, self.corpus.n_items)
                        self.neg_tails[i][j] = tails[i]
                    else:  # sample negative tail
                        head = heads[i] if item_item_relation else self.neg_tails[i][j]
                        tail = self.neg_tails[i][j] if item_item_relation else vals[i]
                        while (head, relations[i], tail) in self.corpus.triplet_set:
                            self.neg_tails[i][j] = np.random.randint(1, self.corpus.n_items)
                            head = heads[i] if item_item_relation else self.neg_tails[i][j]
                            tail = self.neg_tails[i][j] if item_item_relation else vals[i]
                        self.neg_heads[i][j] = heads[i]


class RelationalDynamicAggregation(nn.Module):
    def __init__(self, n_relation, n_freq, relation_embeddings, include_val, device):
        super().__init__()
        self.relation_embeddings = relation_embeddings
        self.include_val = include_val
        self.freq_real = nn.Embedding(n_relation, n_freq)
        self.freq_imag = nn.Embedding(n_relation, n_freq)
        freq = np.linspace(0, 1, n_freq) / 2.
        self.freqs = torch.from_numpy(np.concatenate((freq, -freq))).to(device).float()
        self.relation_range = torch.from_numpy(np.arange(n_relation)).to(device)

    def idft_decay(self, delta_t):
        real, imag = self.freq_real(self.relation_range), self.freq_imag(self.relation_range)
        # create conjugate symmetric to ensure real number output
        x_real = torch.cat([real, real], dim=-1)
        x_imag = torch.cat([imag, -imag], dim=-1)
        w = 2. * np.pi * self.freqs * delta_t.unsqueeze(-1)  # B * H * n_freq
        real_part = w.cos()[:, :, None, :] * x_real[None, None, :, :]  # B * H * R * n_freq
        imag_part = w.sin()[:, :, None, :] * x_imag[None, None, :, :]
        decay = (real_part - imag_part).mean(dim=-1) / 2.  # B * H * R
        return decay.float()

    def forward(self, seq, delta_t_n, target, target_value, valid_mask):
        r_vectors = self.relation_embeddings(self.relation_range)  # R * V
        if self.include_val:
            rv_vectors = r_vectors[None, None, :, :] + target_value
            ri_vectors = rv_vectors * target[:, :, None, :]  # B * -1 * R * V
        else:
            ri_vectors = r_vectors[None, None, :, :] * target[:, :, None, :]  # B * -1 * R * V
        attention = (seq[:, None, :, None, :] * ri_vectors[:, :, None, :, :]).sum(-1)  # B * -1 * H * R
        # shift masked softmax
        attention = attention - attention.max()
        attention = attention.masked_fill(valid_mask == 0, -np.inf).softmax(dim=-2)
        # temporal evolution
        decay = self.idft_decay(delta_t_n).clamp(0, 1).unsqueeze(1).masked_fill(valid_mask==0, 0.)  # B * 1 * H * R
        attention = attention * decay
        # attentional aggregation of history items
        context = (seq[:, None, :, None, :] * attention[:, :, :, :, None]).sum(-3)  # B * -1 * R * V
        return context
