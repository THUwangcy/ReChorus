# -*- coding: UTF-8 -*-

import torch
import torch.nn as nn
import numpy as np

from utils import layers
from models.BaseModel import SequentialModel
from helpers.KDAReader import KDAReader


class FourierTA(SequentialModel):
    reader = 'SeqReader'
    runner = 'BaseRunner'
    extra_log_args = ['t_scalar']

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--emb_size', type=int, default=64,
                            help='Size of embedding vectors.')
        parser.add_argument('--t_scalar', type=int, default=60,
                            help='Time interval scalar.')
        return SequentialModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        super().__init__(args, corpus)
        self.freq_dim = args.emb_size
        self.emb_size = args.emb_size
        self.t_scalar = args.t_scalar
        self._define_params()
        self.apply(self.init_weights)

    def _define_params(self):
        self.user_embeddings = nn.Embedding(self.user_num, self.emb_size)
        self.item_embeddings = nn.Embedding(self.item_num, self.emb_size)

        self.fourier_attn = FourierTemporalAttention(self.emb_size, self.freq_dim, self.device)
        self.W1 = nn.Linear(self.emb_size, self.emb_size)
        self.W2 = nn.Linear(self.emb_size, self.emb_size)
        self.dropout_layer = nn.Dropout(self.dropout)
        self.layer_norm = nn.LayerNorm(self.emb_size)
        self.item_bias = nn.Embedding(self.item_num, 1)

    def forward(self, feed_dict):
        self.check_list = []
        u_ids = feed_dict['user_id']  # B
        i_ids = feed_dict['item_id']  # B * -1
        history = feed_dict['history_items']  # B * H
        delta_t_n = feed_dict['history_delta_t'].float()  # B * H
        batch_size, seq_len = history.shape

        u_vectors = self.user_embeddings(u_ids)
        i_vectors = self.item_embeddings(i_ids)
        his_vectors = self.item_embeddings(history)  # B * H * V

        valid_mask = (history > 0).view(batch_size, 1, seq_len)
        context = self.fourier_attn(his_vectors, delta_t_n, i_vectors, valid_mask)  # B * -1 * V

        residual = context
        # feed forward
        context = self.W1(context)
        context = self.W2(context.relu())
        # dropout, residual and layer_norm
        context = self.dropout_layer(context)
        context = self.layer_norm(residual + context)
        # context = self.layer_norm(context)

        i_bias = self.item_bias(i_ids).squeeze(-1)
        prediction = ((u_vectors[:, None, :] + context) * i_vectors).sum(dim=-1)
        prediction = prediction + i_bias
        out_dict = {'prediction': prediction}
        return out_dict

    class Dataset(SequentialModel.Dataset):
        def _get_feed_dict(self, index):
            feed_dict = super()._get_feed_dict(index)
            delta_t = self.data['time'][index] - feed_dict['history_times']
            feed_dict['history_delta_t'] = KDAReader.norm_time(delta_t, self.model.t_scalar)
            return feed_dict


class FourierTemporalAttention(nn.Module):
    def __init__(self, emb_size: int, freq_dim: int, device):
        super().__init__()
        self.d = emb_size
        self.d_f = freq_dim

        self.freq_real = nn.Parameter(torch.zeros(self.d_f))
        self.freq_imag = nn.Parameter(torch.zeros(self.d_f))
        self.A = nn.Linear(self.d, 10)
        self.A_out = nn.Linear(10, 1, bias=False)

        nn.init.normal_(self.freq_real.data, mean=0.0, std=0.01)
        nn.init.normal_(self.freq_imag.data, mean=0.0, std=0.01)
        freq = np.linspace(0, 1, self.d_f) / 2.
        self.freqs = torch.from_numpy(np.concatenate((freq, -freq))).to(device).float()

    def idft_decay(self, delta_t):
        # create conjugate symmetric to ensure real number output
        x_real = torch.cat([self.freq_real, self.freq_real], dim=-1)
        x_imag = torch.cat([self.freq_imag, -self.freq_imag], dim=-1)
        w = 2. * np.pi * self.freqs * delta_t.unsqueeze(-1)  # B * H * n_freq
        real_part = w.cos() * x_real[None, None, :]  # B * H * n_freq
        imag_part = w.sin() * x_imag[None, None, :]
        decay = (real_part - imag_part).mean(dim=-1) / 2.  # B * H
        return decay.clamp(0, 1).float()

    def forward(self, seq, delta_t_n, target, valid_mask):
        query_vector = seq[:, None, :, :] * target[:, :, None, :]
        attention = self.A_out(self.A(query_vector).tanh()).squeeze(-1)  # B * -1 * H
        # attention = torch.matmul(target, seq.transpose(-2, -1)) / self.d ** 0.5  # B * -1 * H
        # shift masked softmax
        attention = attention - attention.max()
        attention = attention.masked_fill(valid_mask==0, -np.inf).softmax(dim=-1)
        # temporal evolution
        decay = self.idft_decay(delta_t_n).unsqueeze(1).masked_fill(valid_mask==0, 0.)  # B * 1 * H
        attention = attention * decay
        # attentional aggregation of history items
        context = torch.matmul(attention, seq)  # B * -1 * V
        return context
