# -*- coding: UTF-8 -*-
# @Author  : Chenyang Wang
# @Email   : THUwangcy@gmail.com

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from models.BaseModel import GeneralModel


class DisCo(GeneralModel):
    extra_log_args = ['emb_size', 'scale']

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--emb_size', type=int, default=64,
                            help='Size of embedding vectors.')
        parser.add_argument('--scale', type=float, default=1,
                            help='Coefficient of the disentangle loss.')
        return GeneralModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        self.emb_size = args.emb_size
        self.scale = args.scale
        super().__init__(args, corpus)

    def _define_params(self):
        self.encoder = MFEncoder(self.user_num, self.item_num, self.emb_size)
        self.criterion = DisCoLoss(self.emb_size, self.scale)

    def forward(self, feed_dict):
        self.check_list = []
        user, items = feed_dict['user_id'], feed_dict['item_id']
        u_embed, i_embed = self.encoder(user, items)

        prediction = (u_embed[:, None, :] * i_embed).sum(dim=-1)
        out_dict = {'prediction': prediction}

        if feed_dict['phase'] == 'train':
            out_dict.update({
                'z_a': u_embed,
                'z_b': i_embed[:, 0, :]
            })
        return out_dict

    def loss(self, output):
        loss = self.criterion(output['z_a'], output['z_b'])
        return loss

    class Dataset(GeneralModel.Dataset):
        # No need to sample negative items
        def actions_before_epoch(self):
            self.data['neg_items'] = [[] for _ in range(len(self))]


class DisCoLoss(nn.Module):
    def __init__(self, emb_size, scale=1):
        super(DisCoLoss, self).__init__()
        self.scale = scale
        self.emb_size = emb_size
        self.predictor_a = nn.Linear(emb_size, emb_size)
        self.predictor_b = nn.Linear(emb_size, emb_size)
        # self.predictor = nn.Sequential(
        #     nn.Linear(self.emb_size, 128, bias=False), nn.BatchNorm1d(128),
        #     nn.ReLU(inplace=True), nn.Linear(128, self.emb_size, bias=True)
        # )

    @staticmethod
    def similarity(a, b):
        a = F.normalize(a, dim=-1)
        b = F.normalize(b, dim=-1)
        return (1 - (a * b).sum(dim=-1)).mean()

    def forward(self, z_a, z_b):
        """
        Args:
            z_a: user representation of shape [bsz, dim].
            z_b: item representation of shape [bsz, dim].
        Returns:
            A loss scalar.
        """
        loss, bsz, dim = 0, z_a.size(0), z_a.size(1)
        assert dim == self.emb_size
        z_a_proj = self.predictor_a(z_a)
        z_b_proj = self.predictor_b(z_b)

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


class MFEncoder(nn.Module):
    def __init__(self, user_num, item_num, emb_size):
        super(MFEncoder, self).__init__()
        self.user_embedding = nn.Embedding(user_num, emb_size)
        self.item_embedding = nn.Embedding(item_num, emb_size)

    def forward(self, user_id, item_ids):
        u_embed = self.user_embedding(user_id)
        i_embed = self.item_embedding(item_ids)
        return u_embed, i_embed