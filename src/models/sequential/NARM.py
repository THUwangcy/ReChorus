# -*- coding: UTF-8 -*-
# @Author  : Chenyang Wang
# @Email   : THUwangcy@gmail.com

""" NARM
Reference:
    "Neural Attentive Session-based Recommendation"
    Jing Li et al., CIKM'2017.
CMD example:
    python main.py --model_name NARM --emb_size 64 --hidden_size 100 --attention_size 4 --lr 1e-3 --l2 1e-4 \
    --history_max 20 --dataset 'Grocery_and_Gourmet_Food'
"""

import torch
import torch.nn as nn

from models.BaseModel import SequentialModel


class NARM(SequentialModel):
    reader = 'SeqReader'
    runner = 'BaseRunner'
    extra_log_args = ['emb_size', 'hidden_size', 'attention_size']

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--emb_size', type=int, default=64,
                            help='Size of embedding vectors.')
        parser.add_argument('--hidden_size', type=int, default=100,
                            help='Size of hidden vectors in GRU.')
        parser.add_argument('--attention_size', type=int, default=50,
                            help='Size of attention hidden space.')
        return SequentialModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        super().__init__(args, corpus)
        self.emb_size = args.emb_size
        self.hidden_size = args.hidden_size
        self.attention_size = args.attention_size
        self._define_params()
        self.apply(self.init_weights)

    def _define_params(self):
        self.i_embeddings = nn.Embedding(self.item_num, self.emb_size)
        self.encoder_g = nn.GRU(input_size=self.emb_size, hidden_size=self.hidden_size, batch_first=True)
        self.encoder_l = nn.GRU(input_size=self.emb_size, hidden_size=self.hidden_size, batch_first=True)
        self.A1 = nn.Linear(self.hidden_size, self.attention_size, bias=False)
        self.A2 = nn.Linear(self.hidden_size, self.attention_size, bias=False)
        self.attention_out = nn.Linear(self.attention_size, 1, bias=False)
        self.out = nn.Linear(2 * self.hidden_size, self.emb_size, bias=False)

    def forward(self, feed_dict):
        self.check_list = []
        i_ids = feed_dict['item_id']  # [batch_size, -1]
        history = feed_dict['history_items']  # [batch_size, history_max]
        lengths = feed_dict['lengths']  # [batch_size]

        # Embedding Layer
        i_vectors = self.i_embeddings(i_ids)
        his_vectors = self.i_embeddings(history)

        # Encoding Layer
        sort_his_lengths, sort_idx = torch.topk(lengths, k=len(lengths))
        sort_his_vectors = his_vectors.index_select(dim=0, index=sort_idx)
        history_packed = nn.utils.rnn.pack_padded_sequence(sort_his_vectors, sort_his_lengths.cpu(), batch_first=True)
        _, hidden_g = self.encoder_g(history_packed, None)
        output_l, hidden_l = self.encoder_l(history_packed, None)
        output_l, _ = torch.nn.utils.rnn.pad_packed_sequence(output_l, batch_first=True)
        unsort_idx = torch.topk(sort_idx, k=len(lengths), largest=False)[1]
        output_l = output_l.index_select(dim=0, index=unsort_idx)  # [batch_size, history_max, emb_size]
        hidden_g = hidden_g[-1].index_select(dim=0, index=unsort_idx)  # [batch_size, emb_size]

        # Attention Layer
        attention_g = self.A1(hidden_g)
        attention_l = self.A2(output_l)
        attention_value = self.attention_out((attention_g[:, None, :] + attention_l).sigmoid())
        mask = (history > 0).unsqueeze(-1)
        attention_value = attention_value.masked_fill(mask == 0, 0)
        c_l = (attention_value * output_l).sum(1)

        # Prediction Layer
        pred_vector = self.out(torch.cat((hidden_g, c_l), dim=1))
        prediction = (pred_vector[:, None, :] * i_vectors).sum(dim=-1)
        return {'prediction': prediction.view(feed_dict['batch_size'], -1)}
