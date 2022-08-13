# -*- coding: UTF-8 -*-
# @Author  : Chenyang Wang
# @Email   : THUwangcy@gmail.com

""" DirectAU
Reference:
    "Towards Representation Alignment and Uniformity in Collaborative Filtering"
    Wang et al., KDD'2022.
CMD example:
    python main.py --model_name DirectAU --dataset Grocery_and_Gourmet_Food \
                   --emb_size 64 --lr 1e-3 --l2 1e-6 --epoch 500 --gamma 0.3
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.BaseModel import GeneralModel


class DirectAU(GeneralModel):
    reader = 'BaseReader'
    runner = 'BaseRunner'
    extra_log_args = ['emb_size', 'gamma']

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--emb_size', type=int, default=64,
                            help='Size of embedding vectors.')
        parser.add_argument('--gamma', type=float, default=1,
                            help='Weight of the uniformity loss.')
        return GeneralModel.parse_model_args(parser)

    @staticmethod
    def init_weights(m):
        if 'Linear' in str(type(m)):
            nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.normal_(m.bias.data)
        elif 'Embedding' in str(type(m)):
            nn.init.xavier_normal_(m.weight.data)

    def __init__(self, args, corpus):
        super().__init__(args, corpus)
        self.emb_size = args.emb_size
        self.gamma = args.gamma
        self._define_params()
        self.apply(self.init_weights)

    def _define_params(self):
        self.u_embeddings = nn.Embedding(self.user_num, self.emb_size)
        self.i_embeddings = nn.Embedding(self.item_num, self.emb_size)

    @staticmethod
    def alignment(x, y):
        x, y = F.normalize(x, dim=-1), F.normalize(y, dim=-1)
        return (x - y).norm(p=2, dim=1).pow(2).mean()

    @staticmethod
    def uniformity(x):
        x = F.normalize(x, dim=-1)
        return torch.pdist(x, p=2).pow(2).mul(-2).exp().mean().log()

    def forward(self, feed_dict):
        self.check_list = []
        user, items = feed_dict['user_id'], feed_dict['item_id']

        user_e = self.u_embeddings(user)
        item_e = self.i_embeddings(items)

        prediction = (user_e[:, None, :] * item_e).sum(dim=-1)  # [batch_size, -1]
        out_dict = {'prediction': prediction}

        if feed_dict['phase'] == 'train':
            out_dict.update({
                'user_e': user_e,
                'item_e': item_e.squeeze(1)
            })

        return out_dict

    def loss(self, output):
        user_e, item_e = output['user_e'], output['item_e']

        align = self.alignment(user_e, item_e)
        uniform = (self.uniformity(user_e) + self.uniformity(item_e)) / 2
        loss = align + self.gamma * uniform

        return loss

    class Dataset(GeneralModel.Dataset):
        # No need to sample negative items
        def actions_before_epoch(self):
            self.data['neg_items'] = [[] for _ in range(len(self))]
