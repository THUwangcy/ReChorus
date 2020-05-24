# -*- coding: UTF-8 -*-

import torch
import numpy as np

from models.BaseModel import BaseModel
from utils import utils


class GRU4Rec(BaseModel):
    @staticmethod
    def parse_model_args(parser, model_name='GRU4Rec'):
        parser.add_argument('--emb_size', type=int, default=64,
                            help='Size of embedding vectors.')
        parser.add_argument('--hidden_size', type=int, default=200,
                            help='Size of hidden vectors in GRU.')
        return BaseModel.parse_model_args(parser, model_name)

    def __init__(self, args, corpus):
        self.item_num = corpus.n_items
        self.emb_size = args.emb_size
        self.hidden_size = args.hidden_size
        BaseModel.__init__(self, model_path=args.model_path)

    def _define_params(self):
        self.i_embeddings = torch.nn.Embedding(self.item_num, self.emb_size)
        self.rnn = torch.nn.GRU(input_size=self.emb_size, hidden_size=self.hidden_size, batch_first=True)
        self.out = torch.nn.Linear(self.hidden_size, self.emb_size, bias=False)
        self.embeddings = ['i_embeddings']

    def forward(self, feed_dict):
        self.check_list, self.embedding_l2 = [], []
        i_ids = feed_dict['item_id']           # [batch_size, -1]
        history = feed_dict['history_items']   # [batch_size, history_max]
        lengths = feed_dict['lengths']         # [batch_size]
        batch_size = feed_dict['batch_size']

        i_vectors = self.i_embeddings(i_ids)
        his_vectors = self.i_embeddings(history)
        self.embedding_l2.extend([i_vectors, his_vectors])

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

        out_dict = {'prediction': prediction.view(batch_size, -1), 'check': self.check_list}
        return out_dict

    def get_feed_dict(self, corpus, data, batch_start, batch_size, phase):
        feed_dict = BaseModel.get_feed_dict(self, corpus, data, batch_start, batch_size, phase)
        real_batch_size = feed_dict['batch_size']
        history_items = data['item_his'][batch_start: batch_start + real_batch_size].values
        lengths = data['his_length'][batch_start: batch_start + real_batch_size].values
        feed_dict['history_items'] = utils.numpy_to_torch(utils.pad_lst(history_items))
        feed_dict['lengths'] = utils.numpy_to_torch(lengths)
        return feed_dict

    def prepare_batches(self, corpus, data, batch_size, phase):
        data = data[data['his_length'] > 0]  # history length must be at least 1
        return BaseModel.prepare_batches(self, corpus, data, batch_size, phase)
