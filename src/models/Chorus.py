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
    def parse_model_args(parser, model_name='Chorus'):
        parser.add_argument('--stage', type=int, default=2,
                            help='Stage of training: 1-KG pretrain, 2-recommendation.')
        parser.add_argument('--lr_scale', type=float, default=0.1,
                            help='Scale the lr for parameters in pretrained KG model.')
        parser.add_argument('--margin', type=float, default=1,
                            help='Margin in hinge loss.')
        parser.add_argument('--base_method', type=str, default='BPR',
                            help='Basic method to generate recommendations: BPR, GMF')
        return SLRC.parse_model_args(parser, model_name)

    def __init__(self, args, corpus):
        self.emb_size = args.emb_size
        self.time_scalar = args.time_scalar
        self.stage = args.stage
        self.margin = args.margin
        self.base_method = args.base_method
        self.kg_lr = args.lr_scale * args.lr

        assert self.stage in [1, 2]
        self.pretrain_path = '../model/KG/KG__{}__emb_size={}__margin={}.pt'\
            .format(corpus.dataset, self.emb_size, self.margin)
        if self.stage == 1:
            args.model_path = self.pretrain_path

        SLRC.__init__(self, args, corpus)
        self.relation_range = utils.numpy_to_torch(np.arange(self.relation_num))

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
        self.embeddings = ['u_embeddings', 'i_embeddings', 'r_embeddings',
                           'user_bias', 'item_bias', 'betas', 'mus', 'sigmas']

        self.kg_loss = torch.nn.MarginRankingLoss(margin=self.margin)

    def actions_before_train(self):
        if self.stage == 2:
            if os.path.exists(self.pretrain_path):
                self.load_model(self.pretrain_path)
            else:
                raise ValueError('Pretrained KG model does not exist, please run with "--stage 1"')

    def forward(self, feed_dict):
        self.check_list, self.embedding_l2 = [], []

        if self.stage == 1 and feed_dict['phase'] == 'train':
            prediction = self.kg_forward(feed_dict)
        else:
            prediction = self.rec_forward(feed_dict)

        out_dict = {'prediction': prediction, 'check': self.check_list}
        return out_dict

    def kernel_functions(self, r_interval, betas, sigmas, mus):
        """
        Define kernel function for each relation (exponential distribution by default)
        :return [batch_size, -1, relation_num]
        """
        decay_lst = []
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
        batch_size = feed_dict['batch_size']

        u_vectors = self.u_embeddings(u_ids)
        i_vectors = self.i_embeddings(i_ids)
        self.embedding_l2.extend([u_vectors, i_vectors])

        # Temporal Kernel Function
        betas = (self.betas(c_ids) + 1).clamp(min=1e-10, max=10)
        sigmas = (self.sigmas(c_ids) + 1).clamp(min=1e-10, max=10)
        mus = self.mus(c_ids) + 1
        mask = (r_interval >= 0).double()  # mask positions where there is no corresponding relational history
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
        return prediction.view(batch_size, -1)

    def kg_forward(self, feed_dict):
        head_ids = feed_dict['head_id']          # [batch_size]
        tail_ids = feed_dict['tail_id']          # [batch_size]
        relation_ids = feed_dict['relation_id']  # [batch_size]

        head_vectors = self.i_embeddings(head_ids)
        tail_vectors = self.i_embeddings(tail_ids)
        relation_vectors = self.r_embeddings(relation_ids)
        self.embedding_l2.extend([head_vectors, tail_vectors, relation_vectors])

        # TransE
        prediction = -((head_vectors + relation_vectors - tail_vectors) ** 2).sum(-1)
        return prediction

    def loss(self, feed_dict, predictions):
        if self.stage == 1:
            real_batch_size = feed_dict['batch_size']
            pos_pred, neg_pred = predictions[:real_batch_size * 2], predictions[real_batch_size * 2:]
            loss = self.kg_loss(pos_pred, neg_pred, utils.numpy_to_torch(np.ones(real_batch_size * 2)))
        else:
            loss = BaseModel.loss(self, feed_dict, predictions)
        return loss

    def get_feed_dict(self, corpus, data, batch_start, batch_size, phase):
        batch_end = min(len(data), batch_start + batch_size)
        real_batch_size = batch_end - batch_start
        if self.stage == 1 and phase == 'train':
            feed_dict = self.kg_feed_dict(corpus, data, batch_start, real_batch_size)
        else:
            feed_dict = SLRC.get_feed_dict(self, corpus, data, batch_start, real_batch_size, phase)
        feed_dict['phase'] = phase
        return feed_dict

    def prepare_batches(self, corpus, data, batch_size, phase):
        if self.stage == 1 and phase == 'train':
            data = corpus.relation_df.sample(frac=1).reset_index(drop=True)
        return BaseModel.prepare_batches(self, corpus, data, batch_size, phase)

    def kg_feed_dict(self, corpus, data, batch_start, real_batch_size):
        head_ids = data['head'][batch_start: batch_start + real_batch_size].values
        tail_ids = data['tail'][batch_start: batch_start + real_batch_size].values
        relation_ids = data['relation'][batch_start: batch_start + real_batch_size].values
        neg_tails = np.random.randint(1, self.item_num, size=real_batch_size)
        neg_heads = np.random.randint(1, self.item_num, size=real_batch_size)
        for i in range(real_batch_size):
            while (head_ids[i], relation_ids[i], neg_tails[i]) in corpus.triplet_set:
                neg_tails[i] = np.random.randint(1, self.item_num)
            while (neg_heads[i], relation_ids[i], tail_ids[i]) in corpus.triplet_set:
                neg_heads[i] = np.random.randint(1, self.item_num)
        head_ids = np.concatenate((head_ids, head_ids, head_ids, neg_heads))
        tail_ids = np.concatenate((tail_ids, tail_ids, neg_tails, tail_ids))
        relation_ids = np.tile(relation_ids, 4)

        # the head and tail are reversed because the relations we want are is_complement_of, is_substitute_of,
        # which are reversed in terms of the original also_buy, also_view
        feed_dict = {
            'head_id': utils.numpy_to_torch(tail_ids),          # [batch_size]
            'tail_id': utils.numpy_to_torch(head_ids),          # [batch_size]
            'relation_id': utils.numpy_to_torch(relation_ids),  # [batch_size]
            'batch_size': real_batch_size
        }
        return feed_dict

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
            return BaseModel.customize_parameters(self)
