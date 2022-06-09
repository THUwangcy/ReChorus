# -*- coding: UTF-8 -*-
# @Author  : Chenyang Wang
# @Email   : THUwangcy@gmail.com

""" Caser
Reference:
    "Personalized Top-N Sequential Recommendation via Convolutional Sequence Embedding"
    Jiaxi Tang et al., WSDM'2018.
Reference code:
    https://github.com/graytowne/caser_pytorch
Note:
    We use a maximum of L (instead of history_max) horizontal filters to prevent excessive CNN layers.
    Besides, to keep consistent with other sequential models, we do not use the sliding window to generate
    training instances in the paper, and set the parameter T as 1.
CMD example:
    python main.py --model_name Caser --emb_size 64 --L 5 --num_horizon 64 --num_vertical 32 --lr 1e-3 --l2 1e-4 \
    --history_max 20 --dataset 'Grocery_and_Gourmet_Food'
"""

import torch
from torch import nn
import torch.nn.functional as F

from models.BaseModel import SequentialModel


class Caser(SequentialModel):
    reader = 'SeqReader'
    runner = 'BaseRunner'
    extra_log_args = ['emb_size', 'num_horizon', 'num_vertical', 'L']

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--emb_size', type=int, default=64,
                            help='Size of embedding vectors.')
        parser.add_argument('--num_horizon', type=int, default=16,
                            help='Number of horizon convolution kernels.')
        parser.add_argument('--num_vertical', type=int, default=8,
                            help='Number of vertical convolution kernels.')
        parser.add_argument('--L', type=int, default=4,
                            help='Union window size.')
        return SequentialModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        super().__init__(args, corpus)
        self.emb_size = args.emb_size
        self.max_his = args.history_max
        self.num_horizon = args.num_horizon
        self.num_vertical = args.num_vertical
        self.l = args.L
        assert self.l <= self.max_his  # use L instead of max_his to avoid excessive conv_h
        self._define_params()
        self.apply(self.init_weights)

    def _define_params(self):
        self.u_embeddings = nn.Embedding(self.user_num, self.emb_size)
        self.i_embeddings = nn.Embedding(self.item_num, self.emb_size, padding_idx=0)
        lengths = [i + 1 for i in range(self.l)]
        self.conv_h = nn.ModuleList(
            [nn.Conv2d(in_channels=1, out_channels=self.num_horizon, kernel_size=(i, self.emb_size)) for i in lengths])
        self.conv_v = nn.Conv2d(in_channels=1, out_channels=self.num_vertical, kernel_size=(self.max_his, 1))

        self.fc_dim_h = self.num_horizon * len(lengths)
        self.fc_dim_v = self.num_vertical * self.emb_size
        fc_dim_in = self.fc_dim_v + self.fc_dim_h
        self.fc = nn.Linear(fc_dim_in, self.emb_size)
        self.out = nn.Linear(self.emb_size * 2, self.emb_size)


    def forward(self, feed_dict):
        self.check_list = []
        u_ids = feed_dict['user_id']
        i_ids = feed_dict['item_id']  # [batch_size, -1]
        history = feed_dict['history_items']  # [batch_size, history_max]
        batch_size, seq_len = history.shape

        pad_len = self.max_his - seq_len
        history = F.pad(history, [0, pad_len])
        his_vectors = self.i_embeddings(history).unsqueeze(1)  # [batch_size, 1, history_max, emb_size]

        # Convolution Layers
        out, out_h, out_v = None, None, None
        # vertical conv layer
        if self.num_vertical > 0:
            out_v = self.conv_v(his_vectors)
            out_v = out_v.view(-1, self.fc_dim_v)  # prepare for fully connect
        # horizontal conv layer
        out_hs = list()
        if self.num_horizon > 0:
            for conv in self.conv_h:
                conv_out = conv(his_vectors).squeeze(3).relu()
                pool_out = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
                out_hs.append(pool_out)
            out_h = torch.cat(out_hs, 1)  # prepare for fully connect

        # Fully-connected Layers
        user_vector = self.u_embeddings(u_ids)
        z = self.fc(torch.cat([out_v, out_h], 1)).relu()
        his_vector = self.out(torch.cat([z, user_vector], 1))

        i_vectors = self.i_embeddings(i_ids)
        prediction = (his_vector[:, None, :] * i_vectors).sum(-1)
        return {'prediction': prediction.view(batch_size, -1)}