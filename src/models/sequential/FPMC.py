# -*- coding: UTF-8 -*-
# @Author  : Chenyang Wang
# @Email   : THUwangcy@gmail.com

""" FPMC
Reference:
    "Factorizing Personalized Markov Chains for Next-Basket Recommendation"
    Rendle et al., WWW'2010.
CMD example:
    python main.py --model_name FPMC --emb_size 64 --lr 1e-3 --l2 1e-6 --history_max 20 \
    --dataset 'Grocery_and_Gourmet_Food'
"""

import torch
import torch.nn as nn
import numpy as np

from models.BaseModel import SequentialModel


class FPMC(SequentialModel):
    reader = 'SeqReader'
    runner = 'BaseRunner'
    extra_log_args = ['emb_size']

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--emb_size', type=int, default=64,
                            help='Size of embedding vectors.')
        return SequentialModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        super().__init__(args, corpus)
        self.emb_size = args.emb_size
        self._define_params()
        self.apply(self.init_weights)

    def _define_params(self):
        self.ui_embeddings = nn.Embedding(self.user_num, self.emb_size)
        self.iu_embeddings = nn.Embedding(self.item_num, self.emb_size)
        self.li_embeddings = nn.Embedding(self.item_num, self.emb_size)
        self.il_embeddings = nn.Embedding(self.item_num, self.emb_size)

    def forward(self, feed_dict):
        self.check_list = []
        u_id = feed_dict['user_id']  # [batch_size]
        i_ids = feed_dict['item_id']  # [batch_size, -1]
        li_id = feed_dict['last_item_id']  # [batch_size]

        ui_vectors = self.ui_embeddings(u_id)
        iu_vectors = self.iu_embeddings(i_ids)
        li_vectors = self.li_embeddings(li_id)
        il_vectors = self.il_embeddings(i_ids)

        prediction = (ui_vectors[:, None, :] * iu_vectors).sum(-1) + (li_vectors[:, None, :] * il_vectors).sum(-1)
        return {'prediction': prediction.view(feed_dict['batch_size'], -1)}

    class Dataset(SequentialModel.Dataset):
        def _get_feed_dict(self, index):
            user_id, target_item = self.data['user_id'][index], self.data['item_id'][index]
            if self.phase != 'train' and self.model.test_all:
                neg_items = np.arange(1, self.corpus.n_items)
            else:
                neg_items = self.data['neg_items'][index]
            item_ids = np.concatenate([[target_item], neg_items]).astype(int)
            pos = self.data['position'][index]
            last_item_id = self.corpus.user_his[user_id][pos - 1][0]
            feed_dict = {
                'user_id': user_id,
                'item_id': item_ids,
                'last_item_id': last_item_id
            }
            return feed_dict
