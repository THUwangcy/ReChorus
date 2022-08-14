# -*- coding: UTF-8 -*-
# @Author  : Zhefan Wang
# @Email   : wzf19@mails.tsinghua.edu.cn

import torch
import torch.nn as nn

from models.BaseModel import SequentialModel


class YouTube(SequentialModel):
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
        self.i_embeddings = nn.Embedding(self.item_num, self.emb_size)
        self.layer = nn.Linear(self.emb_size, self.emb_size)

    def forward(self, feed_dict):
        self.check_list = []
        i_ids = feed_dict['item_id']  # [batch_size, -1]
        history = feed_dict['history_items']  # [batch_size, history_max]
        lengths = feed_dict['lengths']  # [batch_size]

        his_vectors = self.i_embeddings(history)
        his_vectors_mean = his_vectors.sum(1) / lengths[:, None].float()

        i_vectors = self.i_embeddings(i_ids)
        u_vectors = self.layer(his_vectors_mean)

        prediction = (u_vectors[:, None, :] * i_vectors).sum(-1)

        return {'prediction': prediction.view(feed_dict['batch_size'], -1)}
