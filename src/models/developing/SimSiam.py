# -*- coding: UTF-8 -*-
# @Author  : Chenyang Wang
# @Email   : THUwangcy@gmail.com

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from models.BaseModel import GeneralModel


class SimSiam(GeneralModel):
    extra_log_args = ['emb_size']

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--emb_size', type=int, default=64,
                            help='Size of embedding vectors.')
        return GeneralModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        self.emb_size = args.emb_size
        super().__init__(args, corpus)

    def _define_params(self):
        self.user_online = nn.Embedding(self.user_num, self.emb_size)
        self.item_target = nn.Embedding(self.item_num, self.emb_size)
        self.predictor_user = nn.Linear(self.emb_size, self.emb_size)
        self.predictor_item = nn.Linear(self.emb_size, self.emb_size)

    # 初始化很敏感
    @staticmethod
    def init_weights(m):
        if 'Linear' in str(type(m)):
            nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.normal_(m.bias.data)
        elif 'Embedding' in str(type(m)):
            nn.init.xavier_normal_(m.weight.data)

    def forward(self, feed_dict):
        self.check_list = []
        user, items = feed_dict['user_id'], feed_dict['item_id']

        u_target = self.user_online(user)
        u_online = self.predictor_user(u_target)
        i_target = self.item_target(items)
        i_online = self.predictor_item(i_target)

        # prediction = (self.item_target(items) * self.user_online(user)[:, None, :]).sum(-1)
        prediction = (i_online * u_target[:, None, :]).sum(dim=-1) + \
                     (u_online[:, None, :] * i_target).sum(dim=-1)
        # prediction = (u_online[:, None, :] * i_target).sum(dim=-1)
        out_dict = {'prediction': prediction}

        if feed_dict['phase'] == 'train':
            out_dict.update({
                'u_online': u_online,
                'u_target': u_target,
                'i_online': i_online,
                'i_target': i_target
            })
        return out_dict

    def loss(self, output):
        u_online, u_target = output['u_online'], output['u_target']
        i_online, i_target = output['i_online'], output['i_target']

        u_online = F.normalize(u_online, dim=-1)
        u_target = F.normalize(u_target, dim=-1)
        i_online = F.normalize(i_online, dim=-1)
        i_target = F.normalize(i_target, dim=-1)

        # Euclidean distance between normalized vectors can be replaced with their negative inner product
        loss_ui = 2 - 2 * (u_online * i_target.detach()).sum(dim=-1)
        loss_iu = 2 - 2 * (i_online * u_target.detach()).sum(dim=-1)

        return (loss_ui + loss_iu).mean()

    class Dataset(GeneralModel.Dataset):
        # No need to sample negative items
        def actions_before_epoch(self):
            self.data['neg_items'] = [[] for _ in range(len(self))]
