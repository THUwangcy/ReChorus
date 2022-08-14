# -*- coding: UTF-8 -*-
# @Author  : Zhefan Wang
# @Email   : wzf19@mails.tsinghua.edu.cn

import torch
import torch.nn as nn

from models.BaseModel import SequentialModel


class MIND(SequentialModel):
    extra_log_args = ['emb_size', 'K', 'p', 'iterations', 'relu_layer']

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--emb_size', type=int, default=64,
                            help='Size of embedding vectors.')
        parser.add_argument('--K', type=int, default=2,
                            help='Max number of hidden intent.')
        parser.add_argument('--p', type=str, default='inf',
                            help='A tunable parameter for adjusting the attention distribution.')
        parser.add_argument('--iterations', type=int, default=3,
                            help='The number of iterations for dynamic routing.')
        parser.add_argument('--relu_layer', type=int, default=0,
                            help='The layer after dynamic routing.')
        return SequentialModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        super().__init__(args, corpus)
        self.emb_size = args.emb_size
        self.max_his = args.history_max
        self.K = args.K
        self.p = args.p
        self.relu_layer = args.relu_layer
        if self.p != 'inf':
            self.p = int(self.p)
        self.iterations = args.iterations
        self._define_params()
        self.apply(self.init_weights)

    def _define_params(self):
        self.i_embeddings = nn.Embedding(self.item_num, self.emb_size)
        self.routing = Routing(self.emb_size, self.max_his, self.iterations, self.K, self.relu_layer)

    def label_aware_attention(self, user_capsule, label_item):
        # user_capsule  [batch_size, K, emb_size]
        # label_item  [batch_size, emb_size]
        # valid_capsule  [batch_size, K]
        batch_size, _, __ = user_capsule.shape
        weight = (user_capsule * label_item[:, None, :]).sum(-1)
        if self.p == 'inf':
            idx_select = weight.max(-1)[1]  # bsz
            output = user_capsule[torch.arange(batch_size), idx_select, :]
        else:
            weight = weight ** self.p
            weight = weight.softmax(-1)
            output = (weight[:, :, None] * user_capsule).sum(1)
        return output

    def forward(self, feed_dict):
        self.check_list = []
        i_ids = feed_dict['item_id']  # [batch_size, -1]
        history = feed_dict['history_items']  # [batch_size, history_max]
        batch_size, seq_len = history.shape
        valid_his = (history > 0).long()

        his_vectors = self.i_embeddings(history)
        his_vectors *= valid_his[:, :, None].float()
        i_vectors = self.i_embeddings(i_ids)

        user_capsule = self.routing(his_vectors, valid_his)
        if feed_dict['phase'] == 'train':
            attn_output = self.label_aware_attention(user_capsule, i_vectors[:, 0])  # [batch_size, emb_size]
            prediction = (attn_output[:, None, :] * i_vectors).sum(-1)
        else:
            scores = (user_capsule[:, :, None, :] * i_vectors[:, None, :, :]).sum(-1)  # [batch_size, max_K, -1]
            prediction, _ = torch.max(scores, 1)
        return {'prediction': prediction.view(batch_size, -1)}


class Routing(nn.Module):
    def __init__(self, emb_size, max_his, iterations, K, relu_layer):
        super().__init__()
        self.emb_size = emb_size
        self.max_his = max_his
        self.iterations = iterations
        self.K = K
        self.S = nn.Linear(self.emb_size, self.emb_size, bias=False)
        self.relu_layer = relu_layer
        if self.relu_layer:
            self.layer = nn.Sequential(
                nn.Linear(self.emb_size, self.emb_size),
                nn.ReLU()
            )

    @staticmethod
    def squash(x):
        x_squared_len = (x ** 2).sum(-1, keepdim=True)
        scalar_factor = x_squared_len / (1 + x_squared_len) / torch.sqrt(x_squared_len + 1e-9)
        return x * scalar_factor

    def forward(self, low_capsule, valid_his):
        # low_capsule  [batch_size, seq_len, emb_size]
        batch_size, seq_len, _ = low_capsule.shape
        B = nn.init.normal_(torch.empty(batch_size, self.K, seq_len), mean=0.0, std=1.0).to(low_capsule.device)
        low_capsule_new = self.S(low_capsule)
        low_capsule_new = low_capsule_new.repeat(1, 1, self.K).reshape((-1, seq_len, self.K, self.emb_size))
        low_capsule_new = low_capsule_new.transpose(1, 2)  # [batch_size, K, seq_len, emb_size]
        low_capsule_iter = low_capsule_new.detach()
        for i in range(self.iterations):
            atten_mask = valid_his[:, None, :].repeat(1, self.K, 1)
            paddings = torch.zeros_like(atten_mask).float()
            W = B.softmax(1)  # [batch_size, K, seq_len]
            W = torch.where(atten_mask == 0, paddings, W)
            W = W[:, :, None, :]  # [batch_size, K, 1, seq_len]
            if i + 1 < self.iterations:
                Z = torch.matmul(W, low_capsule_iter)  # [batch_size, K, 1, emb_size]
                U = self.squash(Z)  # [batch_size, K, 1, emb_size]
                delta_B = torch.matmul(low_capsule_iter, U.transpose(2, 3))  # [batch_size, K, seq_len, 1]
                delta_B = delta_B.reshape((-1, self.K, seq_len))
                B += delta_B  # [batch_size, K, seq_len]
            else:
                Z = torch.matmul(W, low_capsule_new)  # [batch_size, K, 1, emb_size]
                U = self.squash(Z)  # [batch_size, K, 1, emb_size]
        U = U.reshape((-1, self.K, self.emb_size))
        if self.relu_layer:
            U = self.layer(U)
        return U