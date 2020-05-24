# -*- coding: UTF-8 -*-

import torch
import numpy as np

from utils import utils
from models.BaseModel import BaseModel


class SLRC(BaseModel):
    @staticmethod
    def parse_model_args(parser, model_name='SLRC'):
        parser.add_argument('--emb_size', type=int, default=64,
                            help='Size of embedding vectors.')
        parser.add_argument('--time_scalar', type=int, default=60 * 60 * 24 * 100,
                            help='Time scalar for time intervals.')
        return BaseModel.parse_model_args(parser, model_name)

    def __init__(self, args, corpus):
        self.emb_size = args.emb_size
        self.time_scalar = args.time_scalar
        self.user_num = corpus.n_users
        self.item_num = corpus.n_items
        self.relation_num = corpus.n_relations
        self.category_num = corpus.item_meta_df['category'].max() + 1
        self.item2cate = dict(zip(corpus.item_meta_df['item_id'].values, corpus.item_meta_df['category'].values))
        BaseModel.__init__(self, model_path=args.model_path)

    def _define_params(self):
        self.u_embeddings = torch.nn.Embedding(self.user_num, self.emb_size)
        self.i_embeddings = torch.nn.Embedding(self.item_num, self.emb_size)
        self.user_bias = torch.nn.Embedding(self.user_num, 1)
        self.item_bias = torch.nn.Embedding(self.item_num, 1)
        self.alphas = torch.nn.Embedding(self.category_num, self.relation_num)
        self.pis = torch.nn.Embedding(self.category_num, self.relation_num)
        self.betas = torch.nn.Embedding(self.category_num, self.relation_num)
        self.sigmas = torch.nn.Embedding(self.category_num, self.relation_num)
        self.mus = torch.nn.Embedding(self.category_num, self.relation_num)
        self.embeddings = ['u_embeddings', 'i_embeddings', 'user_bias', 'item_bias',
                           'alphas', 'pis', 'betas', 'sigmas', 'mus']

    def forward(self, feed_dict):
        self.check_list, self.embedding_l2 = [], []
        u_ids = feed_dict['user_id']                   # [batch_size]
        i_ids = feed_dict['item_id']                   # [batch_size, -1]
        c_ids = feed_dict['category_id']               # [batch_size, -1]
        r_interval = feed_dict['relational_interval']  # [batch_size, -1, relation_num]
        batch_size = feed_dict['batch_size']

        # Excitation
        alphas, pis = self.alphas(c_ids), self.pis(c_ids) + 0.5
        betas, mus = self.betas(c_ids) + 1., self.mus(c_ids) + 1.
        sigmas = (self.sigmas(c_ids) + 1.).clamp(min=1e-10, max=10)
        mask = (r_interval >= 0).double()
        delta_t = r_interval * mask
        norm_dist = torch.distributions.normal.Normal(mus, sigmas)
        exp_dist = torch.distributions.exponential.Exponential(betas)
        decay = pis * exp_dist.log_prob(delta_t).exp() + (1. - pis) * norm_dist.log_prob(delta_t).exp()
        excitation = (alphas * decay * mask).sum(-1)  # [batch_size, -1]

        # Base Intensity (CF)
        u_bias = self.user_bias(u_ids)
        i_bias = self.item_bias(i_ids).squeeze(-1)
        cf_u_vectors = self.u_embeddings(u_ids)
        cf_i_vectors = self.i_embeddings(i_ids)
        self.embedding_l2.extend([cf_u_vectors, cf_i_vectors])
        base_intensity = (cf_u_vectors[:, None, :] * cf_i_vectors).sum(-1)
        base_intensity = base_intensity + u_bias + i_bias

        prediction = base_intensity + excitation

        out_dict = {'prediction': prediction.view(batch_size, -1), 'check': self.check_list}
        return out_dict

    def get_feed_dict(self, corpus, data, batch_start, batch_size, phase):
        batch_end = min(len(data), batch_start + batch_size)
        real_batch_size = batch_end - batch_start
        user_ids = data['user_id'][batch_start: batch_start + real_batch_size].values
        item_ids = data['item_id'][batch_start: batch_start + real_batch_size].values
        history_items = data['item_his'][batch_start: batch_start + real_batch_size].values
        history_times = data['time_his'][batch_start: batch_start + real_batch_size].values
        times = data['time'][batch_start: batch_start + real_batch_size].values

        neg_items = self.get_neg_items(corpus, data, batch_start, real_batch_size, phase)
        item_ids = np.concatenate([np.expand_dims(item_ids, -1), neg_items], axis=1)

        # Find information related to the target item:
        # - category id
        # - time intervals w.r.t. recent relational interactions (-1 if not existing)
        category_ids = np.array([[self.item2cate[x] for x in candidate_lst] for candidate_lst in item_ids])
        relational_intervals = list()
        for r_idx in range(0, self.relation_num):
            intervals = np.ones_like(item_ids) * -1.
            for i, candidate_lst in enumerate(item_ids):
                for j, target_item in enumerate(candidate_lst):
                    for k in range(len(history_items[i]))[::-1]:
                        if (history_items[i][k], r_idx, target_item) in corpus.triplet_set:
                            intervals[i][j] = times[i] - history_times[i][k]
                            break
            relational_intervals.append(intervals)
        relational_intervals = np.stack(relational_intervals, axis=2) / self.time_scalar

        feed_dict = {
            'user_id': utils.numpy_to_torch(user_ids),                          # [batch_size]
            'item_id': utils.numpy_to_torch(item_ids),                          # [batch_size, -1]
            'category_id': utils.numpy_to_torch(category_ids),                  # [batch_size, -1]
            'relational_interval': utils.numpy_to_torch(relational_intervals),  # [batch_size, -1, relation_num]
            'batch_size': real_batch_size
        }
        return feed_dict
