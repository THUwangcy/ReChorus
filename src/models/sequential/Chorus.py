# -*- coding: UTF-8 -*-
# @Author  : Chenyang Wang
# @Email   : THUwangcy@gmail.com

""" Chorus
Reference:
    "Make It a Chorus: Knowledge- and Time-aware Item Modeling for Sequential Recommendation"
    Chenyang Wang et al., SIGIR'2020.
CMD example:
    python main.py --model_name Chorus --emb_size 64 --margin 1 --lr 5e-4 --l2 1e-5 --epoch 50 --early_stop 0 \
    --batch_size 512 --dataset 'Grocery_and_Gourmet_Food' --stage 1
    python main.py --model_name Chorus --emb_size 64 --margin 1 --lr_scale 0.1 --lr 1e-3 --l2 0 \
    --dataset 'Grocery_and_Gourmet_Food' --base_method 'BPR' --stage 2
"""

import os
import torch
import torch.nn as nn
import torch.distributions
import numpy as np

from utils import utils
from models.BaseModel import SequentialModel


class Chorus(SequentialModel):
    reader = 'KGReader'
    runner = 'BaseRunner'
    extra_log_args = ['margin', 'lr_scale', 'stage']

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--stage', type=int, default=2,
                            help='Stage of training: 1-KG_pretrain, 2-recommendation.')
        parser.add_argument('--base_method', type=str, default='BPR',
                            help='Basic method to generate recommendations: BPR, GMF')
        parser.add_argument('--emb_size', type=int, default=64,
                            help='Size of embedding vectors.')
        parser.add_argument('--time_scalar', type=int, default=60 * 60 * 24 * 100,
                            help='Time scalar for time intervals.')
        parser.add_argument('--category_col', type=str, default='i_category',
                            help='The name of category column in item_meta.csv.')
        parser.add_argument('--lr_scale', type=float, default=0.1,
                            help='Scale the lr for parameters in pre-trained KG model.')
        parser.add_argument('--margin', type=float, default=1,
                            help='Margin in hinge loss.')
        return SequentialModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        super().__init__(args, corpus)
        self.margin = args.margin
        self.stage = args.stage
        self.kg_lr = args.lr_scale * args.lr
        self.base_method = args.base_method
        self.emb_size = args.emb_size
        self.time_scalar = args.time_scalar
        self.relations = corpus.item_relations
        self.relation_num = len(corpus.item_relations) + 1
        if args.category_col in corpus.item_meta_df.columns:
            self.category_col = args.category_col
            self.category_num = corpus.item_meta_df[self.category_col].max() + 1
        else:
            self.category_col, self.category_num = None, 1  # a virtual global category
        self._define_params()
        self.apply(self.init_weights)

        assert self.stage in [1, 2]
        self.pretrain_path = '../model/Chorus/KG__{}__emb_size={}__margin={}.pt' \
            .format(corpus.dataset, self.emb_size, self.margin)
        if self.stage == 1:
            self.model_path = self.pretrain_path
        if self.stage == 2:
            if os.path.exists(self.pretrain_path):
                self.load_model(self.pretrain_path)
            else:
                raise ValueError('Pre-trained KG model does not exist, please run with "--stage 1"')
        self.relation_range = torch.from_numpy(np.arange(self.relation_num)).to(self.device)

    def _define_params(self):
        self.u_embeddings = nn.Embedding(self.user_num, self.emb_size)
        self.i_embeddings = nn.Embedding(self.item_num, self.emb_size)
        self.r_embeddings = nn.Embedding(self.relation_num, self.emb_size)
        self.betas = nn.Embedding(self.category_num, self.relation_num)
        self.mus = nn.Embedding(self.category_num, self.relation_num)
        self.sigmas = nn.Embedding(self.category_num, self.relation_num)
        self.prediction = nn.Linear(self.emb_size, 1, bias=False)
        self.user_bias = nn.Embedding(self.user_num, 1)
        self.item_bias = nn.Embedding(self.item_num, 1)

        self.kg_loss = nn.MarginRankingLoss(margin=self.margin)

    def forward(self, feed_dict):
        self.check_list = []
        if self.stage == 1 and feed_dict['phase'] == 'train':
            prediction = self.kg_forward(feed_dict)
        else:
            prediction = self.rec_forward(feed_dict)
        return {'prediction': prediction}

    def kernel_functions(self, r_interval, betas, sigmas, mus):
        """
        Define kernel function for each relation (exponential distribution by default)
        :return [batch_size, -1, relation_num]
        """
        decay_lst = list()
        for r_idx in range(0, self.relation_num):
            delta_t = r_interval[:, :, r_idx]
            beta, sigma, mu = betas[:, :, r_idx], sigmas[:, :, r_idx], mus[:, :, r_idx]
            if r_idx > 0 and 'complement' in self.relations[r_idx - 1]:
                norm_dist = torch.distributions.normal.Normal(0, beta)
                decay = norm_dist.log_prob(delta_t).exp()
            elif r_idx > 0 and 'substitute' in self.relations[r_idx - 1]:
                neg_norm_dist = torch.distributions.normal.Normal(0, beta)
                norm_dist = torch.distributions.normal.Normal(mu, sigma)
                decay = -neg_norm_dist.log_prob(delta_t).exp() + norm_dist.log_prob(delta_t).exp()
            else:  # exponential by default
                exp_dist = torch.distributions.exponential.Exponential(beta, validate_args=False)
                decay = exp_dist.log_prob(delta_t).exp()
            decay_lst.append(decay.clamp(-1, 1))
        return torch.stack(decay_lst, dim=2)

    def rec_forward(self, feed_dict):
        u_ids = feed_dict['user_id']  # [batch_size]
        i_ids = feed_dict['item_id']  # [batch_size, -1]
        c_ids = feed_dict['category_id']  # [batch_size, -1]
        r_interval = feed_dict['relational_interval']  # [batch_size, -1, relation_num]

        u_vectors = self.u_embeddings(u_ids)
        i_vectors = self.i_embeddings(i_ids)

        # Temporal Kernel Function
        betas = (self.betas(c_ids) + 1).clamp(min=1e-10, max=10)
        sigmas = (self.sigmas(c_ids) + 1).clamp(min=1e-10, max=10)
        mus = self.mus(c_ids) + 1
        mask = (r_interval >= 0).float()  # mask positions where there is no corresponding relational history
        temporal_decay = self.kernel_functions(r_interval * mask, betas, sigmas, mus)
        temporal_decay = temporal_decay * mask  # [batch_size, -1, relation_num]

        # Dynamic Integrations
        r_vectors = self.r_embeddings(self.relation_range)
        ri_vectors = i_vectors[:, :, None, :] + r_vectors[None, None, :, :]  # [batch_size, -1, relation_num, emb_size]
        chorus_vectors = i_vectors + (temporal_decay[:, :, :, None] * ri_vectors).sum(2)  # [batch_size, -1, emb_size]

        # Prediction
        if self.base_method.upper().strip() == 'GMF':
            mf_vector = u_vectors[:, None, :] * chorus_vectors
            prediction = self.prediction(mf_vector)
        else:
            u_bias = self.user_bias(u_ids)
            i_bias = self.item_bias(i_ids).squeeze(-1)
            prediction = (u_vectors[:, None, :] * chorus_vectors).sum(-1)
            prediction = prediction + u_bias + i_bias
        return prediction.view(feed_dict['batch_size'], -1)

    def kg_forward(self, feed_dict):
        head_ids = feed_dict['head_id']  # [batch_size, 4]
        tail_ids = feed_dict['tail_id']  # [batch_size, 4]
        relation_ids = feed_dict['relation_id']  # [batch_size, 4]

        head_vectors = self.i_embeddings(head_ids)
        tail_vectors = self.i_embeddings(tail_ids)
        relation_vectors = self.r_embeddings(relation_ids)

        # TransE
        prediction = -((head_vectors + relation_vectors - tail_vectors) ** 2).sum(-1)
        return prediction

    def loss(self, out_dict):
        if self.stage == 1:
            predictions = out_dict['prediction']
            batch_size = predictions.shape[0]
            pos_pred, neg_pred = predictions[:, :2].flatten(), predictions[:, 2:].flatten()
            target = torch.from_numpy(np.ones(batch_size * 2, dtype=np.float32)).to(self.device)
            loss = self.kg_loss(pos_pred, neg_pred, target)
        else:
            loss = super().loss(out_dict)
        return loss

    def customize_parameters(self):
        if self.stage == 2:
            weight_p, kg_p, bias_p = [], [], []
            for name, p in filter(lambda x: x[1].requires_grad, self.named_parameters()):
                if 'bias' in name:
                    bias_p.append(p)
                elif 'i_embeddings' in name or 'r_embeddings' in name:
                    kg_p.append(p)
                else:
                    weight_p.append(p)
            optimize_dict = [
                {'params': weight_p},
                {'params': kg_p, 'lr': self.kg_lr},  # scale down the lr of pretrained embeddings
                {'params': bias_p, 'weight_decay': 0.0}
            ]
            return optimize_dict
        else:
            return super().customize_parameters()

    class Dataset(SequentialModel.Dataset):
        def __init__(self, model, corpus, phase):
            super().__init__(model, corpus, phase)
            self.kg_train = self.model.stage == 1 and self.phase == 'train'
            if self.kg_train:
                self.data = utils.df_to_dict(self.corpus.relation_df)
                self.neg_heads = np.zeros(len(self), dtype=int)
                self.neg_tails = np.zeros(len(self), dtype=int)
            else:
                col_name = self.model.category_col
                items = self.corpus.item_meta_df['item_id']
                categories = self.corpus.item_meta_df[col_name] if col_name is not None else np.zeros_like(items)
                self.item2cate = dict(zip(items, categories))

        def _get_feed_dict(self, index):
            if self.kg_train:
                head, tail = self.data['head'][index], self.data['tail'][index]
                relation = self.data['relation'][index]
                head_id = np.array([head, head, head, self.neg_heads[index]])
                tail_id = np.array([tail, tail, self.neg_tails[index], tail])
                relation_id = np.array([relation] * 4)
                feed_dict = {'head_id': tail_id, 'tail_id': head_id, 'relation_id': relation_id}
                # ↑ the head and tail are reversed due to the relations we want are is_complement_of, is_substitute_of,
                # which are opposite to the original relations also_buy, also_view
            else:
                # Collect information related to the target item:
                # - category id
                # - time intervals w.r.t. recent relational interactions (-1 if not existing)
                feed_dict = super()._get_feed_dict(index)
                user_id, time = self.data['user_id'][index], self.data['time'][index]
                history_item, history_time = feed_dict['history_items'], feed_dict['history_times']
                category_id = [self.item2cate[x] for x in feed_dict['item_id']]
                relational_interval = list()
                for i, target_item in enumerate(feed_dict['item_id']):
                    interval = np.ones(self.model.relation_num, dtype=float) * -1
                    # relational intervals
                    for r_idx in range(1, self.model.relation_num):
                        for j in range(len(history_item))[::-1]:
                            if (history_item[j], r_idx, target_item) in self.corpus.triplet_set:
                                interval[r_idx] = (time - history_time[j]) / self.model.time_scalar
                                break
                    relational_interval.append(interval)
                feed_dict['category_id'] = np.array(category_id)
                feed_dict['relational_interval'] = np.array(relational_interval, dtype=np.float32)
            return feed_dict

        def actions_before_epoch(self):
            if self.kg_train:  # sample negative heads and tails for the KG embedding task
                for i in range(len(self)):
                    head, tail, relation = self.data['head'][i], self.data['tail'][i], self.data['relation'][i]
                    self.neg_tails[i] = np.random.randint(1, self.corpus.n_items)
                    self.neg_heads[i] = np.random.randint(1, self.corpus.n_items)
                    while (head, relation, self.neg_tails[i]) in self.corpus.triplet_set:
                        self.neg_tails[i] = np.random.randint(1, self.corpus.n_items)
                    while (self.neg_heads[i], relation, tail) in self.corpus.triplet_set:
                        self.neg_heads[i] = np.random.randint(1, self.corpus.n_items)
            else:
                super().actions_before_epoch()
