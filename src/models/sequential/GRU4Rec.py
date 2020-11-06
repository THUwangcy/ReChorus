# -*- coding: UTF-8 -*-

import torch
import torch.nn as nn
import numpy as np

from models.BaseModel import BaseModel


class GRU4Rec(BaseModel):
    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--emb_size', type=int, default=64,
                            help='Size of embedding vectors.')
        parser.add_argument('--hidden_size', type=int, default=100,
                            help='Size of hidden vectors in GRU.')
        return BaseModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        self.emb_size = args.emb_size
        self.hidden_size = args.hidden_size
        super().__init__(args, corpus)

    def _define_params(self):
        self.i_embeddings = nn.Embedding(self.item_num, self.emb_size)
        self.rnn = nn.GRU(input_size=self.emb_size, hidden_size=self.hidden_size, batch_first=True)
        self.out = nn.Linear(self.hidden_size, self.emb_size, bias=False)

    def forward(self, feed_dict):
        self.check_list = []
        i_ids = feed_dict['item_id']  # [batch_size, -1]
        history = feed_dict['history_items']  # [batch_size, history_max]
        lengths = feed_dict['lengths']  # [batch_size]

        i_vectors = self.i_embeddings(i_ids)
        his_vectors = self.i_embeddings(history)

        # Sort and Pack
        sort_his_lengths, sort_idx = torch.topk(lengths, k=len(lengths))
        sort_his_vectors = his_vectors.index_select(dim=0, index=sort_idx)
        history_packed = torch.nn.utils.rnn.pack_padded_sequence(sort_his_vectors, sort_his_lengths, batch_first=True)

        # RNN
        output, hidden = self.rnn(history_packed, None)

        # Unsort
        sort_rnn_vector = self.out(hidden[-1])
        unsort_idx = torch.topk(sort_idx, k=len(lengths), largest=False)[1]
        rnn_vector = sort_rnn_vector.index_select(dim=0, index=unsort_idx)

        # Predicts
        prediction = (rnn_vector[:, None, :] * i_vectors).sum(-1)
        return prediction.view(feed_dict['batch_size'], -1)

    class Dataset(BaseModel.Dataset):
        def _prepare(self):
            idx_select = np.array(self.data['his_length']) > 0  # history length must be non-zero
            for key in self.data:
                self.data[key] = np.array(self.data[key])[idx_select]
            super()._prepare()

        def _get_feed_dict(self, index):
            feed_dict = super()._get_feed_dict(index)
            feed_dict['history_items'] = np.array(self.data['item_his'][index])
            feed_dict['lengths'] = self.data['his_length'][index]
            return feed_dict
