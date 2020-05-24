# -*- coding: UTF-8 -*-

import torch

from utils import utils
from models.BaseModel import BaseModel


class BPR(BaseModel):
    @staticmethod
    def parse_model_args(parser, model_name='BPR'):
        parser.add_argument('--emb_size', type=int, default=64,
                            help='Size of embedding vectors.')
        return BaseModel.parse_model_args(parser, model_name)

    def __init__(self, args, corpus):
        self.emb_size = args.emb_size
        BaseModel.__init__(self, args, corpus)

    def _define_params(self):
        self.u_embeddings = torch.nn.Embedding(self.user_num, self.emb_size)
        self.i_embeddings = torch.nn.Embedding(self.item_num, self.emb_size)
        self.user_bias = torch.nn.Embedding(self.user_num, 1)
        self.item_bias = torch.nn.Embedding(self.item_num, 1)
        self.embeddings = ['u_embeddings', 'i_embeddings', 'user_bias', 'item_bias']

    def forward(self, feed_dict):
        self.check_list, self.embedding_l2 = [], []
        u_ids = feed_dict['user_id']  # [batch_size]
        i_ids = feed_dict['item_id']  # [batch_size, n_candidates]

        cf_u_vectors = self.u_embeddings(u_ids)
        cf_i_vectors = self.i_embeddings(i_ids)
        u_bias = self.user_bias(u_ids)
        i_bias = self.item_bias(i_ids).squeeze(-1)
        self.embedding_l2.extend([cf_u_vectors, cf_i_vectors])

        prediction = (cf_u_vectors[:, None, :] * cf_i_vectors).sum(dim=-1)
        prediction = prediction + u_bias + i_bias

        out_dict = {'prediction': prediction.view(feed_dict['batch_size'], -1), 'check': self.check_list}
        return out_dict

    def get_feed_dict(self, corpus, data, batch_start, batch_size, phase):
        feed_dict = BaseModel.get_feed_dict(self, corpus, data, batch_start, batch_size, phase)
        real_batch_size = feed_dict['batch_size']
        user_ids = data['user_id'][batch_start: batch_start + real_batch_size].values
        feed_dict['user_id'] = utils.numpy_to_torch(user_ids)  # [batch_size]
        return feed_dict
