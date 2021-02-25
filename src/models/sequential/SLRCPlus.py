# -*- coding: UTF-8 -*-
# @Author  : Chenyang Wang
# @Email   : THUwangcy@gmail.com

""" SLRC+
Reference:
    "Modeling Item-specific Temporal Dynamics of Repeat Consumption for Recommender Systems"
    Chenyang Wang et al., TheWebConf'2019.
Reference code:
    The authors' tensorflow implementation https://github.com/THUwangcy/SLRC
Note:
    We generalize the original SLRC by also including mutual-excitation of relational history interactions.
    This makes SLRC+ a knowledge-aware model, and the original SLRC can be seen that there is only one special
    relation between items and themselves (i.e., repeat consumption).
CMD example:
    python main.py --model_name SLRCPlus --emb_size 64 --lr 5e-4 --l2 1e-5 --dataset 'Grocery_and_Gourmet_Food'
"""

import torch
import torch.nn as nn
import torch.distributions
import numpy as np

from models.BaseModel import SequentialModel
from helpers.KGReader import KGReader


class SLRCPlus(SequentialModel):
    reader = 'KGReader'
    extra_log_args = ['emb_size']

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--emb_size', type=int, default=64,
                            help='Size of embedding vectors.')
        parser.add_argument('--time_scalar', type=int, default=60 * 60 * 24 * 100,
                            help='Time scalar for time intervals.')
        return SequentialModel.parse_model_args(parser)

    def __init__(self, args, corpus: KGReader):
        self.emb_size = args.emb_size
        self.time_scalar = args.time_scalar
        self.relation_num = len(corpus.item_relations) + 1
        super().__init__(args, corpus)

    def _define_params(self):
        self.u_embeddings = nn.Embedding(self.user_num, self.emb_size)
        self.i_embeddings = nn.Embedding(self.item_num, self.emb_size)
        self.user_bias = nn.Embedding(self.user_num, 1)
        self.item_bias = nn.Embedding(self.item_num, 1)

        self.global_alpha = nn.Parameter(torch.tensor(0.))
        self.alphas = nn.Embedding(self.item_num, self.relation_num)
        self.pis = nn.Embedding(self.item_num, self.relation_num)
        self.betas = nn.Embedding(self.item_num, self.relation_num)
        self.sigmas = nn.Embedding(self.item_num, self.relation_num)
        self.mus = nn.Embedding(self.item_num, self.relation_num)

    def forward(self, feed_dict):
        self.check_list = []
        u_ids = feed_dict['user_id']  # [batch_size]
        i_ids = feed_dict['item_id']  # [batch_size, -1]
        r_intervals = feed_dict['relational_interval']  # [batch_size, -1, relation_num]

        # Excitation
        alphas = self.global_alpha + self.alphas(i_ids)
        pis, mus = self.pis(i_ids) + 0.5, self.mus(i_ids) + 1
        betas = (self.betas(i_ids) + 1).clamp(min=1e-10, max=10)
        sigmas = (self.sigmas(i_ids) + 1).clamp(min=1e-10, max=10)
        mask = (r_intervals >= 0).float()
        delta_t = r_intervals * mask
        norm_dist = torch.distributions.normal.Normal(mus, sigmas)
        exp_dist = torch.distributions.exponential.Exponential(betas)
        decay = pis * exp_dist.log_prob(delta_t).exp() + (1 - pis) * norm_dist.log_prob(delta_t).exp()
        excitation = (alphas * decay * mask).sum(-1)  # [batch_size, -1]

        # Base Intensity (MF)
        u_bias = self.user_bias(u_ids)
        i_bias = self.item_bias(i_ids).squeeze(-1)
        cf_u_vectors = self.u_embeddings(u_ids)
        cf_i_vectors = self.i_embeddings(i_ids)
        base_intensity = (cf_u_vectors[:, None, :] * cf_i_vectors).sum(-1)
        base_intensity = base_intensity + u_bias + i_bias

        prediction = base_intensity + excitation
        return {'prediction': prediction.view(feed_dict['batch_size'], -1)}

    class Dataset(SequentialModel.Dataset):
        def _get_feed_dict(self, index):
            feed_dict = super()._get_feed_dict(index)
            user_id, time = self.data['user_id'][index], self.data['time'][index]
            history_item, history_time = feed_dict['history_items'], feed_dict['history_times']

            # Collect time information related to the target item:
            # - re-consuming time gaps
            # - time intervals w.r.t. recent relational interactions
            relational_interval = list()
            for i, target_item in enumerate(feed_dict['item_id']):
                interval = np.ones(self.model.relation_num, dtype=float) * -1  # -1 if not existing
                # the first dimension for re-consuming time gaps
                for j in range(len(history_item))[::-1]:
                    if history_item[j] == target_item:
                        interval[0] = (time - history_time[j]) / self.model.time_scalar
                        break
                # the rest for relational time intervals
                for r_idx in range(1, self.model.relation_num):
                    for j in range(len(history_item))[::-1]:
                        if (history_item[j], r_idx, target_item) in self.corpus.triplet_set:
                            interval[r_idx] = (time - history_time[j]) / self.model.time_scalar
                            break
                relational_interval.append(interval)
            feed_dict['relational_interval'] = np.array(relational_interval, dtype=np.float32)
            return feed_dict
