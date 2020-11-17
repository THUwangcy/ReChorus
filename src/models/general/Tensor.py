# -*- coding: UTF-8 -*-

import torch.nn as nn

from models.general.BPR import BPR


class Tensor(BPR):
    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--time_bin', type=int, default=100,
                            help='Number of time bins.')
        return BPR.parse_model_args(parser)

    def __init__(self, args, corpus):
        self.time_bin = args.time_bin
        self.min_time = corpus.all_df['time'].min()
        self.max_time = corpus.all_df['time'].max()
        self.time_bin_width = (self.max_time - self.min_time + 1.) / self.time_bin
        super().__init__(args, corpus)

    def _define_params(self):
        super()._define_params()
        self.u_t_embeddings = nn.Embedding(self.time_bin, self.emb_size)
        self.i_t_embeddings = nn.Embedding(self.time_bin, self.emb_size)

    def forward(self, feed_dict):
        self.check_list = []
        u_ids = feed_dict['user_id']  # [batch_size]
        t_ids = feed_dict['time_id']  # [batch_size]
        i_ids = feed_dict['item_id']  # [batch_size, -1]

        cf_u_vectors = self.u_embeddings(u_ids)
        cf_i_vectors = self.i_embeddings(i_ids)
        u_t_vectors = self.u_t_embeddings(t_ids)
        i_t_vectors = self.i_t_embeddings(t_ids)
        u_bias = self.user_bias(u_ids)
        i_bias = self.item_bias(i_ids).squeeze()

        prediction = ((cf_u_vectors + i_t_vectors)[:, None, :] * cf_i_vectors +
                      (cf_u_vectors * u_t_vectors)[:, None, :]).sum(dim=-1)
        prediction = prediction + u_bias + i_bias
        return {'prediction': prediction.view(feed_dict['batch_size'], -1)}

    class Dataset(BPR.Dataset):
        def _get_feed_dict(self, index):
            feed_dict = super()._get_feed_dict(index)
            time_ids = (self.data['time'][index] - self.model.min_time) // self.model.time_bin_width
            feed_dict['time_id'] = int(time_ids)
            return feed_dict
