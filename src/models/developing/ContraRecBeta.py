# -*- coding: UTF-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.BaseModel import SequentialModel
from utils import layers


class ContraRecBeta(SequentialModel):
    extra_log_args = ['encoder']

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--emb_size', type=int, default=64,
                            help='Size of embedding vectors.')
        parser.add_argument('--rec_temp', type=float, default=0.2,
                            help='Temperature in recommendation contrastive loss.')
        parser.add_argument('--encoder', type=str, default='BERT4Rec',
                            help='Choose a sequence encoder: GRU4Rec, Caser, BERT4Rec.')
        return SequentialModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        self.emb_size = args.emb_size
        self.max_his = args.history_max
        self.rec_temperature = args.rec_temp
        self.encoder_name = args.encoder
        super().__init__(args, corpus)

    def _define_params(self):
        self.i_embeddings = nn.Embedding(self.item_num, self.emb_size)
        if self.encoder_name == 'GRU4Rec':
            self.encoder = GRU4RecEncoder(self.emb_size, hidden_size=128)
        elif self.encoder_name == 'Caser':
            self.encoder = CaserEncoder(self.emb_size, self.max_his, num_horizon=16, num_vertical=8, l=5)
        elif self.encoder_name == 'BERT4Rec':
            self.encoder = BERT4RecEncoder(self.emb_size, self.max_his, self.device, num_layers=2, num_heads=2)
        else:
            raise ValueError('Invalid sequence encoder.')
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, feed_dict):
        self.check_list = []
        i_ids = feed_dict['item_id']  # bsz, n_candidate
        history = feed_dict['history_items']  # bsz, history_max
        lengths = feed_dict['lengths']  # bsz

        his_vectors = self.i_embeddings(history)
        his_vector = self.encoder(his_vectors, lengths)
        i_vectors = self.i_embeddings(i_ids)
        # his_vector = F.normalize(his_vector, dim=-1)  # ！
        # i_vectors = F.normalize(i_vectors, dim=-1)
        prediction = (his_vector[:, None, :] * i_vectors).sum(-1)
        out_dict = {'prediction': prediction}
        return out_dict

    def loss(self, out_dict):
        predictions = out_dict['prediction'] / self.rec_temperature
        # pre_softmax = (predictions - predictions.max()).softmax(dim=1)
        # rec_loss = - self.rec_temperature * pre_softmax[:, 0].log().mean()
        target = torch.zeros(predictions.shape[0], dtype=torch.int64).to(self.device)
        rec_loss = self.rec_temperature * self.criterion(predictions, target).mean()
        # return super().loss(out_dict)
        return rec_loss


""" Encoder Layers """
class GRU4RecEncoder(nn.Module):
    def __init__(self, emb_size, hidden_size=128):
        super().__init__()
        self.rnn = nn.GRU(input_size=emb_size, hidden_size=hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, emb_size, bias=False)

    def forward(self, seq, lengths):
        # Sort and Pack
        sort_lengths, sort_idx = torch.topk(lengths, k=len(lengths))
        sort_seq = seq.index_select(dim=0, index=sort_idx)
        seq_packed = torch.nn.utils.rnn.pack_padded_sequence(sort_seq, sort_lengths, batch_first=True)

        # RNN
        output, hidden = self.rnn(seq_packed, None)

        # Unsort
        sort_rnn_vector = self.out(hidden[-1])
        unsort_idx = torch.topk(sort_idx, k=len(lengths), largest=False)[1]
        rnn_vector = sort_rnn_vector.index_select(dim=0, index=unsort_idx)

        return rnn_vector

class CaserEncoder(nn.Module):
    def __init__(self, emb_size, max_his, num_horizon=16, num_vertical=8, l=5):
        super().__init__()
        self.max_his = max_his
        lengths = [i + 1 for i in range(l)]
        self.conv_h = nn.ModuleList(
            [nn.Conv2d(in_channels=1, out_channels=num_horizon, kernel_size=(i, emb_size)) for i in lengths])
        self.conv_v = nn.Conv2d(in_channels=1, out_channels=num_vertical, kernel_size=(max_his, 1))
        self.fc_dim_h = num_horizon * len(lengths)
        self.fc_dim_v = num_vertical * emb_size
        fc_dim_in = self.fc_dim_v + self.fc_dim_h
        self.fc = nn.Linear(fc_dim_in, emb_size)

    def forward(self, seq, lengths):
        batch_size, seq_len = seq.size(0), seq.size(1)
        pad_len = self.max_his - seq_len
        seq = F.pad(seq, [0, 0, 0, pad_len]).unsqueeze(1)

        # Convolution Layers
        out_v = self.conv_v(seq).view(-1, self.fc_dim_v)
        out_hs = list()
        for conv in self.conv_h:
            conv_out = conv(seq).squeeze(3).relu()
            pool_out = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
            out_hs.append(pool_out)
        out_h = torch.cat(out_hs, 1)

        # Fully-connected Layers
        his_vector = self.fc(torch.cat([out_v, out_h], 1))
        return his_vector

class BERT4RecEncoder(nn.Module):
    def __init__(self, emb_size, max_his, device, num_layers=2, num_heads=2):
        super().__init__()
        self.device = device

        self.p_embeddings = nn.Embedding(max_his + 1, emb_size)
        self.transformer_block = nn.ModuleList([
            layers.TransformerLayer(d_model=emb_size, d_ff=emb_size, n_heads=num_heads)
            for _ in range(num_layers)
        ])

    def forward(self, seq, lengths):
        batch_size, seq_len = seq.size(0), seq.size(1)
        len_range = torch.from_numpy(np.arange(seq_len)).to(self.device)
        valid_mask = len_range[None, :] < lengths[:, None]

        # Position embedding
        position = len_range[None, :] * valid_mask.long()
        pos_vectors = self.p_embeddings(position)
        seq = seq + pos_vectors

        # Self-attention
        attn_mask = valid_mask.view(batch_size, 1, 1, seq_len)
        for block in self.transformer_block:
            seq = block(seq, attn_mask)
        seq = seq * valid_mask[:, :, None].float()

        his_vector = seq[torch.arange(batch_size), lengths - 1]
        return his_vector
