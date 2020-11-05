# -*- coding: UTF-8 -*-

import os
import torch
import numpy as np

from utils import utils
from models.BaseModel import BaseModel
from models.SLRC import SLRC


class Chorus(SLRC):
    extra_log_args = ['margin', 'lr_scale', 'stage']

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--stage', type=int, default=2,
                            help='Stage of training: 1-KG pretrain, 2-recommendation.')
        parser.add_argument('--lr_scale', type=float, default=0.1,
                            help='Scale the lr for parameters in pretrained KG model.')
        parser.add_argument('--margin', type=float, default=1,
                            help='Margin in hinge loss.')
        parser.add_argument('--base_method', type=str, default='BPR',
                            help='Basic method to generate recommendations: BPR, GMF')
        return SLRC.parse_model_args(parser)

    def __init__(self, args, corpus):
        self.margin = args.margin
        self.stage = args.stage
        self.kg_lr = args.lr_scale * args.lr
        self.base_method = args.base_method
        super().__init__(args, corpus)

        assert self.stage in [1, 2]
        self.pretrain_path = '../model/Chorus/KG__{}__emb_size={}__margin={}.pt' \
            .format(corpus.dataset, self.emb_size, self.margin)
        if self.stage == 1:
            self.model_path = self.pretrain_path
        self.relation_range = torch.from_numpy(np.arange(corpus.n_relations)).to(self.device)

    def actions_before_train(self):
        if self.stage == 2:
            if os.path.exists(self.pretrain_path):
                self.load_model(self.pretrain_path)
            else:
                raise ValueError('Pretrained KG model does not exist, please run with "--stage 1"')

    def _define_params(self):
        self.u_embeddings = torch.nn.Embedding(self.user_num, self.emb_size)
        self.i_embeddings = torch.nn.Embedding(self.item_num, self.emb_size)
        self.r_embeddings = torch.nn.Embedding(self.relation_num, self.emb_size)
        self.betas = torch.nn.Embedding(self.category_num, self.relation_num)
        self.mus = torch.nn.Embedding(self.category_num, self.relation_num)
        self.sigmas = torch.nn.Embedding(self.category_num, self.relation_num)
        self.prediction = torch.nn.Linear(self.emb_size, 1, bias=False)
        self.user_bias = torch.nn.Embedding(self.user_num, 1)
        self.item_bias = torch.nn.Embedding(self.item_num, 1)

        self.kg_loss = torch.nn.MarginRankingLoss(margin=self.margin)

    def forward(self, feed_dict):
        self.check_list = []
        if self.stage == 1 and feed_dict['phase'] == 'train':
            prediction = self.kg_forward(feed_dict)
        else:
            prediction = self.rec_forward(feed_dict)
        return prediction

    def kernel_functions(self, r_interval, betas, sigmas, mus):
        """
        Define kernel function for each relation (exponential distribution by default)
        :return [batch_size, -1, relation_num]
        """
        decay_lst = list()
        for r_idx in range(0, self.relation_num):
            delta_t = r_interval[:, :, r_idx]
            beta, sigma, mu = betas[:, :, r_idx], sigmas[:, :, r_idx], mus[:, :, r_idx]
            if r_idx == 1:  # is_complement_of
                norm_dist = torch.distributions.normal.Normal(0, beta)
                decay = norm_dist.log_prob(delta_t).exp()
            elif r_idx == 2:  # is_substitute_of
                neg_norm_dist = torch.distributions.normal.Normal(0, beta)
                norm_dist = torch.distributions.normal.Normal(mu, sigma)
                decay = -neg_norm_dist.log_prob(delta_t).exp() + norm_dist.log_prob(delta_t).exp()
            else:  # exponential by default
                exp_dist = torch.distributions.exponential.Exponential(beta)
                decay = exp_dist.log_prob(delta_t).exp()
            decay_lst.append(decay.clamp(-1, 1))
        return torch.stack(decay_lst, dim=2)

    def rec_forward(self, feed_dict):
        u_ids = feed_dict['user_id']                   # [batch_size]
        i_ids = feed_dict['item_id']                   # [batch_size, -1]
        c_ids = feed_dict['category_id']               # [batch_size, -1]
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
        head_ids = feed_dict['head_id']          # [batch_size, 4]
        tail_ids = feed_dict['tail_id']          # [batch_size, 4]
        relation_ids = feed_dict['relation_id']  # [batch_size, 4]

        head_vectors = self.i_embeddings(head_ids)
        tail_vectors = self.i_embeddings(tail_ids)
        relation_vectors = self.r_embeddings(relation_ids)

        # TransE
        prediction = -((head_vectors + relation_vectors - tail_vectors) ** 2).sum(-1)
        return prediction

    def loss(self, predictions):
        if self.stage == 1:
            batch_size = predictions.shape[0]
            pos_pred, neg_pred = predictions[:, :2].flatten(), predictions[:, 2:].flatten()
            target = torch.from_numpy(np.ones(batch_size * 2)).to(self.device)
            loss = self.kg_loss(pos_pred, neg_pred, target)
        else:
            loss = super().loss(predictions)
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

    class Dataset(SLRC.Dataset):
        def _prepare(self):
            self.kg_train = self.model.stage == 1 and self.phase == 'train'
            if self.kg_train:
                self.data = utils.df_to_dict(self.corpus.relation_df)
                self.neg_heads = np.zeros(len(self), dtype=int)
                self.neg_tails = np.zeros(len(self), dtype=int)
            super()._prepare()

        def _get_feed_dict(self, index):
            if self.kg_train:
                head, tail = self.data['head'][index], self.data['tail'][index]
                relation = self.data['relation'][index]
                head_id = np.array([head, head, head, self.neg_heads[index]])
                tail_id = np.array([tail, tail, self.neg_tails[index], tail])
                relation_id = np.array([relation] * 4)
                # the head and tail are reversed because the relations we want are is_complement_of, is_substitute_of,
                # which are reversed in terms of the original also_buy, also_view
                feed_dict = {'head_id': tail_id, 'tail_id': head_id, 'relation_id': relation_id}
            else:
                feed_dict = super()._get_feed_dict(index)
            return feed_dict

        def negative_sampling(self):
            if self.kg_train:
                for i in range(len(self)):
                    head, tail, relation = self.data['head'][i], self.data['tail'][i], self.data['relation'][i]
                    self.neg_tails[i] = np.random.randint(1, self.corpus.n_items)
                    self.neg_heads[i] = np.random.randint(1, self.corpus.n_items)
                    while (head, relation, self.neg_tails[i]) in self.corpus.triplet_set:
                        self.neg_tails[i] = np.random.randint(1, self.corpus.n_items)
                    while (self.neg_heads[i], relation, tail) in self.corpus.triplet_set:
                        self.neg_heads[i] = np.random.randint(1, self.corpus.n_items)
            else:
                super().negative_sampling()
