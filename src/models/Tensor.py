# -*- coding: UTF-8 -*-

import torch

from utils import utils
from models.BPR import BPR


class Tensor(BPR):
    @staticmethod
    def parse_model_args(parser, model_name='Tensor'):
        parser.add_argument('--time_bin', type=int, default=100,
                            help='Number of time bins.')
        return BPR.parse_model_args(parser, model_name)

    def __init__(self, args, corpus):
        self.time_bin = args.time_bin
        self.min_time = corpus.min_time
        self.time_bin_width = (corpus.max_time - self.min_time + 1.) / self.time_bin
        BPR.__init__(self, args, corpus)

    def _define_params(self):
        BPR._define_params(self)
        self.u_t_embeddings = torch.nn.Embedding(self.time_bin, self.emb_size)
        self.i_t_embeddings = torch.nn.Embedding(self.time_bin, self.emb_size)
        self.embeddings.extend(['u_t_embeddings', 'i_t_embeddings'])

    def forward(self, feed_dict):
        self.check_list, self.embedding_l2 = [], []
        u_ids = feed_dict['user_id']  # [batch_size]
        t_ids = feed_dict['time_id']  # [batch_size]
        i_ids = feed_dict['item_id']  # [batch_size, -1]

        cf_u_vectors = self.u_embeddings(u_ids)
        cf_i_vectors = self.i_embeddings(i_ids)
        u_t_vectors = self.u_t_embeddings(t_ids)
        i_t_vectors = self.i_t_embeddings(t_ids)
        u_bias = self.user_bias(u_ids)
        i_bias = self.item_bias(i_ids).squeeze()
        self.embedding_l2.extend([cf_u_vectors, cf_i_vectors, u_t_vectors, i_t_vectors])

        prediction = ((cf_u_vectors + i_t_vectors)[:, None, :] * cf_i_vectors +
                      (cf_u_vectors * u_t_vectors)[:, None, :]).sum(dim=-1)
        prediction = prediction + u_bias + i_bias

        out_dict = {'prediction': prediction.view(feed_dict['batch_size'], -1), 'check': self.check_list}
        return out_dict

    def get_feed_dict(self, corpus, data, batch_start, batch_size, phase):
        feed_dict = BPR.get_feed_dict(self, corpus, data, batch_start, batch_size, phase)
        real_batch_size = feed_dict['batch_size']
        times = data['time'][batch_start: batch_start + real_batch_size].values
        time_ids = (times - self.min_time) // self.time_bin_width
        feed_dict['time_id'] = utils.numpy_to_torch(time_ids).long()
        return feed_dict
