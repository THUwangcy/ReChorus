# -*- coding: UTF-8 -*-
# @Author  : Chenyang Wang
# @Email   : THUwangcy@gmail.com

import os
import logging
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from models.BaseModel import SequentialModel
from models.sequential.TiSASRec import TimeIntervalTransformerLayer
from utils import layers


class TiMiRec(SequentialModel):
    runner = 'TiMiRunner'
    extra_log_args = ['emb_size', 'attn_size', 'K', 'temp', 'add_pos', 'predictor', 'n_layers']

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--emb_size', type=int, default=64,
                            help='Size of embedding vectors.')
        parser.add_argument('--attn_size', type=int, default=8,
                            help='Size of attention vectors.')
        parser.add_argument('--K', type=int, default=2,
                            help='Number of hidden intent.')
        parser.add_argument('--add_pos', type=int, default=1,
                            help='Whether add position embedding in extractor.')
        parser.add_argument('--predictor', type=str, default='GRU',
                            help='Choose the main model of the predictor: TiSASRec, SASRec, GRU')
        parser.add_argument('--temp', type=float, default=1,
                            help='Temperature in knowledge distillation loss.')
        parser.add_argument('--n_layers', type=int, default=1,
                            help='Number of the projection layer.')
        parser.add_argument('--stage', type=int, default=3,
                            help='Stage of training: 1-pretrain_extractor, 2-pretrain_predictor, 3-joint_finetune.')
        return SequentialModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        self.emb_size = args.emb_size
        self.attn_size = args.attn_size
        self.K = args.K
        self.add_pos = args.add_pos
        self.predictor = args.predictor
        self.temp = args.temp
        self.n_layers = args.n_layers
        self.stage = args.stage
        self.max_his = args.history_max
        self.max_time = 512  # max time intervals in TiSASRec
        super().__init__(args, corpus)

        if self.stage in [2, 3] and self.predictor == 'TiSASRec':
            setattr(corpus, 'user_min_interval', dict())
            for u, user_df in corpus.all_df.groupby('user_id'):
                time_seqs = user_df['time'].values
                interval_matrix = np.abs(time_seqs[:, None] - time_seqs[None, :])
                min_interval = np.min(interval_matrix + (interval_matrix <= 0) * 0xFFFF)
                corpus.user_min_interval[u] = min_interval

        self.extractor_path = '../model/TiMiRec/Extractor__{}__{}__emb_size={}__K={}__add_pos={}.pt' \
            .format(corpus.dataset, args.random_seed, self.emb_size, self.K, self.add_pos)
        self.predictor_path = '../model/TiMiRec/Predictor__{}__{}__emb_size={}__model={}.pt' \
            .format(corpus.dataset, args.random_seed, self.emb_size, self.predictor)
        if self.stage == 1:
            self.model_path = self.extractor_path
        elif self.stage == 2:
            self.model_path = self.predictor_path

    def _define_params(self):
        if self.stage in [1, 3]:
            self.interest_extractor = MultiInterestExtractor(
                self.K, self.item_num, self.emb_size, self.attn_size, self.max_his, self.add_pos)
        if self.stage in [2, 3]:
            self.intent_predictor = IntentPredictor(
                self.item_num, self.emb_size, self.max_his, self.predictor, self.max_time)
        if self.stage == 3:
            self.proj = nn.Sequential()
            for i, _ in enumerate(range(self.n_layers - 1)):
                self.proj.add_module('proj_' + str(i), nn.Linear(self.emb_size, self.emb_size))
                self.proj.add_module('dropout_' + str(i), nn.Dropout(p=0.5))
                self.proj.add_module('relu_' + str(i), nn.ReLU(inplace=True))
            self.proj.add_module('proj_final', nn.Linear(self.emb_size, self.K))

    def load_model(self, model_path=None):
        if model_path is None:
            model_path = self.model_path
        model_dict = self.state_dict()
        state_dict = torch.load(model_path)
        exist_state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
        model_dict.update(exist_state_dict)
        self.load_state_dict(model_dict)
        logging.info('Load model from ' + model_path)

    def actions_before_train(self):
        if self.stage == 3 and os.path.exists(self.extractor_path):
            self.load_model(self.extractor_path)
            # self.load_model(self.predictor_path)
            return
        logging.info('Train from scratch!')

    @staticmethod
    def similarity(a, b):
        a = F.normalize(a, dim=-1)
        b = F.normalize(b, dim=-1)
        return (a * b).sum(dim=-1)

    @staticmethod
    def js_div(p, q):
        kl = nn.KLDivLoss(reduction='none')
        p, q = p.softmax(-1), q.softmax(-1)
        log_mean = ((p + q) / 2).log()
        js = (kl(log_mean, p) + kl(log_mean, q)) / 2
        return js

    def forward(self, feed_dict):
        self.check_list = []
        i_ids = feed_dict['item_id']  # bsz, -1
        history = feed_dict['history_items']  # bsz, max_his
        t_history, user_min_t = None, None
        if self.stage in [2, 3] and self.predictor == 'TiSASRec':
            t_history = feed_dict['history_times']  # bsz, max_his
            user_min_t = feed_dict['user_min_intervals']  # bsz
        lengths = feed_dict['lengths']  # bsz
        batch_size, seq_len = history.shape

        out_dict = dict()
        if self.stage == 1:  # pretrain extractor
            interest_vectors = self.interest_extractor(history, lengths)  # bsz, K, emb
            i_vectors = self.interest_extractor.i_embeddings(i_ids)
            if feed_dict['phase'] == 'train':
                target_vector = i_vectors[:, 0]  # bsz, emb
                target_intent = (interest_vectors * target_vector[:, None, :]).sum(-1)  # bsz, K
                idx_select = target_intent.max(-1)[1]  # bsz
                user_vector = interest_vectors[torch.arange(batch_size), idx_select, :]  # bsz, emb
                prediction = (user_vector[:, None, :] * i_vectors).sum(-1)
            else:
                prediction = (interest_vectors[:, None, :, :] * i_vectors[:, :, None, :]).sum(-1)  # bsz, -1, K
                prediction = prediction.max(-1)[0]  # bsz, -1
        elif self.stage == 2:  # pretrain predictor
            his_vector = self.intent_predictor(history, lengths, t_history, user_min_t)
            i_vectors = self.intent_predictor.i_embeddings(i_ids)
            prediction = (his_vector[:, None, :] * i_vectors).sum(-1)
        else:  # finetune
            interest_vectors = self.interest_extractor(history, lengths)  # bsz, K, emb
            i_vectors = self.interest_extractor.i_embeddings(i_ids)
            his_vector = self.intent_predictor(history, lengths, t_history, user_min_t)
            pred_intent = self.proj(his_vector)  # bsz, K
            user_vector = (interest_vectors * pred_intent.softmax(-1)[:, :, None]).sum(-2)  # bsz, emb
            if feed_dict['phase'] == 'train':
                # idx_select = pred_intent.max(-1)[1]  # bsz
                # user_vector = interest_vectors[torch.arange(batch_size), idx_select, :]  # bsz, emb
                target_vector = i_vectors[:, 0]  # bsz, emb
                target_intent = (interest_vectors * target_vector[:, None, :]).sum(-1)  # bsz, K
                out_dict['pred_intent'] = pred_intent
                out_dict['target_intent'] = target_intent
                self.check_list.append(('intent', pred_intent.softmax(-1)))
                self.check_list.append(('target', target_intent.softmax(-1)))
            prediction = (user_vector[:, None, :] * i_vectors).sum(-1)
        out_dict['prediction'] = prediction.view(batch_size, -1)

        # For JS divergence analysis
        if self.stage == 3 and feed_dict['phase'] == 'test':
            target_vector = i_vectors[:, 0]  # bsz, emb
            target_intent = self.similarity(interest_vectors, target_vector.unsqueeze(1))  # bsz, K
            out_dict['js'] = self.js_div(target_intent, pred_intent).sum(-1)

        return out_dict

    def loss(self, out_dict: dict):
        if self.stage in [1, 2]:  # pretrain
            loss = super().loss(out_dict)
        else:  # finetune
            pred_intent = out_dict['pred_intent'] / self.temp
            target_intent = out_dict['target_intent'].detach() / self.temp
            kl_criterion = nn.KLDivLoss(reduction='batchmean')
            loss = kl_criterion(F.log_softmax(pred_intent, dim=1), F.softmax(target_intent, dim=1))
            loss = super().loss(out_dict) + self.temp * self.temp * loss
        return loss

    class Dataset(SequentialModel.Dataset):
        def _get_feed_dict(self, index):
            feed_dict = super()._get_feed_dict(index)
            if self.model.stage in [2, 3] and self.model.predictor == 'TiSASRec':
                user_id = self.data['user_id'][index]
                min_interval = self.corpus.user_min_interval[user_id]
                feed_dict['user_min_intervals'] = min_interval
            return feed_dict


class MultiInterestExtractor(nn.Module):
    def __init__(self, k, item_num, emb_size, attn_size, max_his, add_pos):
        super(MultiInterestExtractor, self).__init__()
        self.max_his = max_his
        self.add_pos = add_pos

        self.i_embeddings = nn.Embedding(item_num, emb_size)
        if self.add_pos:
            self.p_embeddings = nn.Embedding(max_his + 1, emb_size)
        self.W1 = nn.Linear(emb_size, attn_size)
        self.W2 = nn.Linear(attn_size, k)
        self.transformer = layers.TransformerLayer(d_model=emb_size, d_ff=emb_size, n_heads=1, kq_same=False)

    def forward(self, history, lengths):
        batch_size, seq_len = history.shape
        valid_his = (history > 0).long()

        his_vectors = self.i_embeddings(history)
        if self.add_pos:
            len_range = torch.from_numpy(np.arange(self.max_his)).to(history.device)
            position = (lengths[:, None] - len_range[None, :seq_len]) * valid_his
            pos_vectors = self.p_embeddings(position)
            his_vectors = his_vectors + pos_vectors

        # Self-attention
        attn_mask = valid_his.view(batch_size, 1, 1, seq_len)
        his_vectors = self.transformer(his_vectors, attn_mask)
        his_vectors = his_vectors * valid_his[:, :, None].float()

        # Multi-Interest Extraction
        attn_score = self.W2(self.W1(his_vectors).tanh())  # bsz, his_max, K
        attn_score = attn_score.masked_fill(valid_his.unsqueeze(-1) == 0, -np.inf)
        attn_score = attn_score.transpose(-1, -2)  # bsz, K, his_max
        attn_score = (attn_score - attn_score.max()).softmax(dim=-1)
        attn_score = attn_score.masked_fill(torch.isnan(attn_score), 0)
        interest_vectors = (his_vectors[:, None, :, :] * attn_score[:, :, :, None]).sum(-2)  # bsz, K, emb
        return interest_vectors


class IntentPredictor(nn.Module):
    def __init__(self, item_num, emb_size, max_his, predictor, max_time):
        super(IntentPredictor, self).__init__()
        self.i_embeddings = nn.Embedding(item_num + 1, emb_size)

        if predictor == 'GRU':
            self.encoder = GRUEncoder(emb_size)
        elif predictor == 'SASRec':
            self.encoder = SASRecEncoder(emb_size, max_his)
        elif predictor == 'TiSASRec':
            self.encoder = TiSASRecEncoder(emb_size, max_his, max_time)
        else:
            raise ValueError('Non-Implemented Predictor.')

    def forward(self, history, lengths, t_history, user_min_t):
        valid_his = (history > 0).long()
        his_vectors = self.i_embeddings(history)
        his_vector = self.encoder(his_vectors, lengths, valid_his, t_history, user_min_t)
        return his_vector


class GRUEncoder(nn.Module):
    def __init__(self, emb_size):
        super().__init__()
        self.rnn = nn.GRU(input_size=emb_size, hidden_size=emb_size, batch_first=True)

    def forward(self, his_vectors, lengths, *args, **kwargs):
        sort_lengths, sort_idx = torch.topk(lengths, k=len(lengths))
        sort_seq = his_vectors.index_select(dim=0, index=sort_idx)
        seq_packed = torch.nn.utils.rnn.pack_padded_sequence(sort_seq, sort_lengths.cpu(), batch_first=True)
        output, hidden = self.rnn(seq_packed, None)
        unsort_idx = torch.topk(sort_idx, k=len(lengths), largest=False)[1]
        his_vector = hidden[-1].index_select(dim=0, index=unsort_idx)
        return his_vector


class SASRecEncoder(nn.Module):
    def __init__(self, emb_size, max_his):
        super().__init__()
        self.max_his = max_his
        self.p_embeddings = nn.Embedding(max_his + 1, emb_size)
        self.transformer = layers.TransformerLayer(d_model=emb_size, d_ff=emb_size, n_heads=1, kq_same=False)

    def forward(self, his_vectors, lengths, valid_his, *args, **kwargs):
        batch_size, seq_len, emb_size = his_vectors.shape
        len_range = torch.from_numpy(np.arange(self.max_his)).to(his_vectors.device)
        position = (lengths[:, None] - len_range[None, :seq_len]) * valid_his
        pos_vectors = self.p_embeddings(position)
        his_vectors = his_vectors + pos_vectors
        attn_mask = valid_his.view(batch_size, 1, 1, seq_len)
        his_vectors = self.transformer(his_vectors, attn_mask)
        his_vectors = his_vectors * valid_his[:, :, None].float()
        # his_vector = his_vectors.sum(1) / lengths[:, None].float()
        his_vector = his_vectors[torch.arange(batch_size), lengths - 1, :]
        return his_vector


class TiSASRecEncoder(nn.Module):
    def __init__(self, emb_size, max_his, max_time):
        super().__init__()
        num_heads, num_layers, dropout = 1, 1, 0
        self.max_his = max_his
        self.max_time = max_time
        self.p_embeddings_ = nn.Embedding(max_his + 1, emb_size)
        self.p_v_embeddings_ = nn.Embedding(max_his + 1, emb_size)
        self.t_embeddings_ = nn.Embedding(max_time + 1, emb_size)
        self.t_v_embeddings_ = nn.Embedding(max_time + 1, emb_size)
        self.ti_transformer_block = nn.ModuleList([
            TimeIntervalTransformerLayer(d_model=emb_size, d_ff=emb_size, n_heads=num_heads,
                                         dropout=dropout, kq_same=False)
            for _ in range(num_layers)
        ])

    def forward(self, his_vectors, lengths, valid_his, t_history, user_min_t, *args, **kwargs):
        batch_size, seq_len, emb_size = his_vectors.shape
        len_range = torch.from_numpy(np.arange(self.max_his)).to(his_vectors.device)
        # Position embedding
        position = (lengths[:, None] - len_range[None, :seq_len]) * valid_his
        pos_k = self.p_embeddings_(position)
        pos_v = self.p_v_embeddings_(position)

        # Interval embedding
        interval_matrix = (t_history[:, :, None] - t_history[:, None, :]).abs()
        interval_matrix = (interval_matrix / user_min_t.view(-1, 1, 1)).long().clamp(0, self.max_time)
        inter_k = self.t_embeddings_(interval_matrix)
        inter_v = self.t_v_embeddings_(interval_matrix)

        causality_mask = np.tril(np.ones((1, 1, seq_len, seq_len), dtype=np.int))
        attn_mask = torch.from_numpy(causality_mask).to(his_vectors.device)

        for block in self.ti_transformer_block:
            his_vectors = block(his_vectors, pos_k, pos_v, inter_k, inter_v, attn_mask)
        his_vectors = his_vectors * valid_his[:, :, None].float()

        # his_vector = his_vectors.sum(1) / lengths[:, None].float()  # average
        his_vector = his_vectors[torch.arange(batch_size), lengths - 1, :]  # use the last/mask item
        return his_vector
