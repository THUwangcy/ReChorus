# -*- coding: UTF-8 -*-

import torch
import numpy as np

from models.GRU4Rec import GRU4Rec


class NARM(GRU4Rec):
    @staticmethod
    def parse_model_args(parser, model_name='NARM'):
        parser.add_argument('--attention_size', type=int, default=50,
                            help='Size of attention hidden space.')
        return GRU4Rec.parse_model_args(parser, model_name)

    def __init__(self, args, corpus):
        self.attention_size = args.attention_size
        GRU4Rec.__init__(self, args, corpus)

    def _define_params(self):
        self.i_embeddings = torch.nn.Embedding(self.item_num, self.emb_size)
        self.encoder_g = torch.nn.GRU(input_size=self.emb_size, hidden_size=self.hidden_size, batch_first=True)
        self.encoder_l = torch.nn.GRU(input_size=self.emb_size, hidden_size=self.hidden_size, batch_first=True)
        self.A1 = torch.nn.Linear(self.hidden_size, self.attention_size, bias=False)
        self.A2 = torch.nn.Linear(self.hidden_size, self.attention_size, bias=False)
        self.attention_out = torch.nn.Linear(self.attention_size, 1, bias=False)
        self.out = torch.nn.Linear(2 * self.hidden_size, self.emb_size, bias=False)
        self.embeddings = ['i_embeddings']

    def forward(self, feed_dict):
        self.check_list, self.embedding_l2 = [], []
        i_ids = feed_dict['item_id']          # [batch_size, -1]
        history = feed_dict['history_items']  # [batch_size, history_max]
        lengths = feed_dict['lengths']        # [batch_size]

        # Embedding Layer
        i_vectors = self.i_embeddings(i_ids)
        his_vectors = self.i_embeddings(history)
        self.embedding_l2.extend([i_vectors, his_vectors])

        # Encoding Layer
        sort_his_lengths, sort_idx = torch.topk(lengths, k=len(lengths))
        sort_his_vectors = his_vectors.index_select(dim=0, index=sort_idx)
        history_packed = torch.nn.utils.rnn.pack_padded_sequence(sort_his_vectors, sort_his_lengths, batch_first=True)
        _, hidden_g = self.encoder_g(history_packed, None)
        output_l, hidden_l = self.encoder_l(history_packed, None)
        output_l, _ = torch.nn.utils.rnn.pad_packed_sequence(output_l, batch_first=True)
        unsort_idx = torch.topk(sort_idx, k=len(lengths), largest=False)[1]
        output_l = output_l.index_select(dim=0, index=unsort_idx)      # [batch_size, history_max, emb_size]
        hidden_g = hidden_g[-1].index_select(dim=0, index=unsort_idx)  # [batch_size, emb_size]

        # Attention Layer
        attention_g = self.A1(hidden_g)
        attention_l = self.A2(output_l)
        attention_value = self.attention_out((attention_g[:, None, :] + attention_l).sigmoid())
        mask = (history == 0).unsqueeze(-1)
        attention_value = attention_value - attention_value.max()
        attention_value = attention_value.masked_fill(mask, -np.inf).softmax(dim=1)
        c_l = (attention_value * output_l).sum(1)

        # Prediction Layer
        pred_vector = self.out(torch.cat((hidden_g, c_l), dim=1))
        prediction = (pred_vector[:, None, :] * i_vectors).sum(dim=-1)

        out_dict = {'prediction': prediction.view(feed_dict['batch_size'], -1), 'check': self.check_list}
        return out_dict
