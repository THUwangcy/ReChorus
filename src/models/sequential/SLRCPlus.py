# -*- coding: UTF-8 -*-

import torch
import torch.nn as nn
import torch.distributions
import numpy as np

from models.BaseModel import GeneralModel
from helpers.KGReader import KGReader

"""
  An enhancement to original SLRC. 
  Also include the excitation of relational history interactions (not only repeat consumptions), 
  and related parameters are category-specific instead of item-specific.
"""
class SLRCPlus(GeneralModel):
    reader = 'KGReader'

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--emb_size', type=int, default=64,
                            help='Size of embedding vectors.')
        parser.add_argument('--time_scalar', type=int, default=60*60*24*100,
                            help='Time scalar for time intervals.')
        parser.add_argument('--category_col', type=str, default='i_category',
                            help='The name of category column in item_meta.csv.')
        return GeneralModel.parse_model_args(parser)

    def __init__(self, args, corpus: KGReader):
        self.emb_size = args.emb_size
        self.time_scalar = args.time_scalar
        self.relation_num = len(corpus.item_relations) + 1
        if args.category_col in corpus.item_meta_df.columns:
            self.category_col = args.category_col
            self.category_num = corpus.item_meta_df[self.category_col].max() + 1
        else:
            self.category_col, self.category_num = None, 1  # a virtual global category
        super().__init__(args, corpus)

    def _define_params(self):
        self.u_embeddings = nn.Embedding(self.user_num, self.emb_size)
        self.i_embeddings = nn.Embedding(self.item_num, self.emb_size)
        self.user_bias = nn.Embedding(self.user_num, 1)
        self.item_bias = nn.Embedding(self.item_num, 1)

        self.alphas = nn.Embedding(self.category_num, self.relation_num)
        self.pis = nn.Embedding(self.category_num, self.relation_num)
        self.betas = nn.Embedding(self.category_num, self.relation_num)
        self.sigmas = nn.Embedding(self.category_num, self.relation_num)
        self.mus = nn.Embedding(self.category_num, self.relation_num)

    def forward(self, feed_dict):
        self.check_list = []
        u_ids = feed_dict['user_id']  # [batch_size]
        i_ids = feed_dict['item_id']  # [batch_size, -1]
        c_ids = feed_dict['category_id']  # [batch_size, -1]
        r_intervals = feed_dict['relational_interval']  # [batch_size, -1, relation_num]

        # Excitation
        alphas, pis, mus = self.alphas(c_ids), self.pis(c_ids) + 0.5, self.mus(c_ids) + 1
        betas = (self.betas(c_ids) + 1).clamp(min=1e-10, max=10)
        sigmas = (self.sigmas(c_ids) + 1).clamp(min=1e-10, max=10)
        mask = (r_intervals >= 0).float()
        delta_t = r_intervals * mask
        norm_dist = torch.distributions.normal.Normal(mus, sigmas)
        exp_dist = torch.distributions.exponential.Exponential(betas)
        decay = pis * exp_dist.log_prob(delta_t).exp() + (1 - pis) * norm_dist.log_prob(delta_t).exp()
        excitation = (alphas * decay * mask).sum(-1)  # [batch_size, -1]

        # Base Intensity (CF)
        u_bias = self.user_bias(u_ids)
        i_bias = self.item_bias(i_ids).squeeze(-1)
        cf_u_vectors = self.u_embeddings(u_ids)
        cf_i_vectors = self.i_embeddings(i_ids)
        base_intensity = (cf_u_vectors[:, None, :] * cf_i_vectors).sum(-1)
        base_intensity = base_intensity + u_bias + i_bias

        prediction = base_intensity + excitation
        return {'prediction': prediction.view(feed_dict['batch_size'], -1)}

    class Dataset(GeneralModel.Dataset):
        def _prepare(self):
            category_col = self.model.category_col
            items = self.corpus.item_meta_df['item_id']
            categories = self.corpus.item_meta_df[category_col] if category_col is not None else np.zeros_like(items)
            self.item2cate = dict(zip(items, categories))
            super()._prepare()

        def _get_feed_dict(self, index):
            feed_dict = super()._get_feed_dict(index)
            user_id, time = self.data['user_id'][index], self.data['time'][index]
            history_item, history_time = self.data['item_his'][index], self.data['time_his'][index]

            # Collect information related to the target item:
            # - category id
            # - time intervals w.r.t. recent relational interactions (-1 if not existing)
            category_id = [self.item2cate[x] for x in feed_dict['item_id']]
            relational_interval = list()
            for i, target_item in enumerate(feed_dict['item_id']):
                interval = np.ones(self.model.relation_num, dtype=float) * -1
                # reserve the first dimension for the repeat consumption interval
                for j in range(len(history_item))[::-1]:
                    if history_item[j] == target_item:
                        interval[0] = (time - history_time[j]) / self.model.time_scalar
                        break
                # the rest for relational intervals
                for r_idx in range(1, self.model.relation_num):
                    for j in range(len(history_item))[::-1]:
                        if (history_item[j], r_idx, target_item) in self.corpus.triplet_set:
                            interval[r_idx] = (time - history_time[j]) / self.model.time_scalar
                            break
                relational_interval.append(interval)

            feed_dict['user_id'] = user_id
            feed_dict['category_id'] = np.array(category_id)
            feed_dict['relational_interval'] = np.array(relational_interval, dtype=np.float32)
            return feed_dict
