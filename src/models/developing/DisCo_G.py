# -*- coding: UTF-8 -*-
# @Author  : Chenyang Wang
# @Email   : THUwangcy@gmail.com

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from models.BaseModel import GeneralModel
from models.general.LightGCN import LightGCN
from models.general.LightGCN import LGCNEncoder


class DisCo_G(GeneralModel):
    extra_log_args = ['emb_size', 'scale']

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--emb_size', type=int, default=64,
                            help='Size of embedding vectors.')
        parser.add_argument('--scale', type=float, default=0.01,
                            help='Coefficient of the disentangle loss.')
        parser.add_argument('--n_layers', type=int, default=3,
                            help='Number of LightGCN layers.')
        return GeneralModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        self.emb_size = args.emb_size
        self.scale = args.scale
        self.n_layers = args.n_layers
        self.norm_adj = LightGCN.build_adjmat(corpus.n_users, corpus.n_items, corpus.train_clicked_set)
        super().__init__(args, corpus)

    def _define_params(self):
        self.encoder = LGCNEncoder(self.user_num, self.item_num, self.emb_size, self.norm_adj, self.n_layers)
        self.criterion = DisCoLoss(self.emb_size, self.scale)

        self.predictor_a = nn.Linear(self.emb_size, self.emb_size)
        self.predictor_b = nn.Linear(self.emb_size, self.emb_size)

    def forward(self, feed_dict):
        self.check_list = []
        user, items = feed_dict['user_id'], feed_dict['item_id']
        u_embed, i_embed = self.encoder(user, items)
        u_embed_proj = self.predictor_a(u_embed)
        i_embed_proj = self.predictor_b(i_embed)

        prediction = (u_embed[:, None, :] * i_embed).sum(dim=-1)
        out_dict = {'prediction': prediction}

        if feed_dict['phase'] == 'train':
            out_dict.update({
                'z_a': u_embed,
                'z_b': i_embed[:, 0],
                'z_a_proj': u_embed_proj,
                'z_b_proj': i_embed_proj[:, 0]
            })
        return out_dict

    def loss(self, output):
        return self.criterion(output['z_a'], output['z_b'], output['z_a_proj'], output['z_b_proj'])

    class Dataset(GeneralModel.Dataset):
        # No need to sample negative items
        def actions_before_epoch(self):
            self.data['neg_items'] = [[] for _ in range(len(self))]


class DisCoLoss(nn.Module):
    def __init__(self, emb_size, scale=1):
        super(DisCoLoss, self).__init__()
        self.scale = scale
        self.emb_size = emb_size

    @staticmethod
    def similarity(a, b):
        a = F.normalize(a, dim=-1)
        b = F.normalize(b, dim=-1)
        return (1 - (a * b).sum(dim=-1)).mean()

    def forward(self, z_a, z_b, z_a_proj, z_b_proj):
        """
        Args:
            z_a: user representation of shape [bsz, dim].
            z_b: item representation of shape [bsz, dim].
        Returns:
            A loss scalar.
        """
        loss, bsz, dim = 0, z_a.size(0), z_a.size(1)
        assert dim == self.emb_size
        # z_a_proj = self.predictor_a(z_a)
        # z_b_proj = self.predictor_b(z_b)

        # Similarity
        loss += (self.similarity(z_a_proj, z_b.detach()) +
                 self.similarity(z_b_proj, z_a.detach())) / 2

        # Disentangle
        z_a, z_b = F.normalize(z_a, dim=-1), F.normalize(z_b, dim=-1)
        eye = torch.eye(dim, dtype=torch.float).to(z_a.device)
        z_a = (z_a - z_a.mean(dim=0)) / z_a.std(dim=0)
        z_b = (z_b - z_b.mean(dim=0)) / z_b.std(dim=0)
        c = (z_a.t() @ z_b) / bsz
        balance = eye + (1 - eye) / (dim - 1)
        c_diff = (c - eye).pow(2) * balance
        dis_loss = c_diff.sum(1).mean()
        loss += self.scale * dis_loss

        return loss
