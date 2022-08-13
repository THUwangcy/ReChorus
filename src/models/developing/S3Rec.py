# -*- coding: UTF-8 -*-

import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.BaseModel import SequentialModel
from utils import layers


class S3Rec(SequentialModel):
    reader = 'SeqReader'
    runner = 'BaseRunner'
    extra_log_args = ['emb_size', 'mip_weight', 'sp_weight', 'mask_ratio', 'stage']

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--emb_size', type=int, default=64,
                            help='Size of embedding vectors.')
        parser.add_argument('--mip_weight', type=float, default=0.2,
                            help='Coefficient of the MIP loss.')
        parser.add_argument('--sp_weight', type=float, default=0.5,
                            help='Coefficient of the SP loss.')
        parser.add_argument('--mask_ratio', type=float, default=0.2,
                            help='Proportion of masked positions in the sequence.')
        parser.add_argument('--stage', type=int, default=1,
                            help='Stage of training: 1-pretrain, 2-finetune, default-from_scratch.')
        return SequentialModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        super().__init__(args, corpus)
        self.emb_size = args.emb_size
        self.mip_weight = args.mip_weight
        self.sp_weight = args.sp_weight
        self.mask_ratio = args.mask_ratio
        self.stage = args.stage
        self.max_his = args.history_max
        self._define_params()
        self.apply(self.init_weights)

        # assert(self.stage in [1, 2])
        self.pre_path = '../model/S3Rec/Pre__{}.pt'.format(corpus.dataset)
        self.model_path = self.pre_path if self.stage == 1 else self.model_path
        if self.stage == 2:  # fine-tune
            if os.path.exists(self.pre_path):
                self.load_model(self.pre_path)
            else:
                logging.info('Train from scratch!')

    def _define_params(self):
        self.i_embeddings = nn.Embedding(self.item_num + 1, self.emb_size, padding_idx=0)
        self.encoder = BERT4RecEncoder(self.emb_size, self.max_his, num_layers=2, num_heads=2, dropout=0.2)
        self.mip_norm = nn.Linear(self.emb_size, self.emb_size)
        self.sp_norm = nn.Linear(self.emb_size, self.emb_size)

    def _masked_item_prediction(self, seq_output, target_item_emb):
        return (self.mip_norm(seq_output)[:, None, :] * target_item_emb).sum(-1).sigmoid().view(-1)  # [B*L]

    def _segment_prediction(self, context, segment_emb):
        return (self.sp_norm(context) * segment_emb).sum(-1).sigmoid()  # [B]

    def forward(self, feed_dict):
        self.check_list = []
        if self.stage == 1 and feed_dict['phase'] == 'train':
            mask_token = self.item_num
            # MIP
            mask_seq, seq_len = feed_dict['mask_seq'], feed_dict['seq_len']
            seq_vectors = self.i_embeddings(mask_seq)
            seq_output = self.encoder(seq_vectors, seq_len)
            pos_vectors = self.i_embeddings(feed_dict['pos_item'])
            neg_vectors = self.i_embeddings(feed_dict['neg_item'])
            pos_score = self._masked_item_prediction(seq_output, pos_vectors)
            neg_score = self._masked_item_prediction(seq_output, neg_vectors)
            mip_distance = torch.sigmoid(pos_score - neg_score)
            valid_mask = torch.arange(mask_seq.size(1)).to(self.device)[None, :] < seq_len[:, None]
            mip_mask = (feed_dict['mask_seq'] == mask_token).float() * valid_mask.float()
            # SP
            seg_seq_vectors = self.i_embeddings(feed_dict['mask_seg_seq'])
            pos_seg_vectors = self.i_embeddings(feed_dict['pos_seg'])
            neg_seg_vectors = self.i_embeddings(feed_dict['neg_seg'])
            segment_context = self.encoder(seg_seq_vectors, seq_len)
            pos_segment_emb = self.encoder(pos_seg_vectors, seq_len)
            neg_segment_emb = self.encoder(neg_seg_vectors, seq_len)
            pos_segment_score = self._segment_prediction(segment_context, pos_segment_emb)
            neg_segment_score = self._segment_prediction(segment_context, neg_segment_emb)
            sp_distance = torch.sigmoid(pos_segment_score - neg_segment_score)
            out_dict = {'mip_dis': mip_distance, 'mip_mask': mip_mask, 'sp_dis': sp_distance}
        else:
            i_ids = feed_dict['item_id']  # bsz, n_candidate
            history = feed_dict['history_items']  # bsz, history_max
            lengths = feed_dict['lengths']  # bsz
            his_vectors = self.i_embeddings(history)
            his_vector = self.encoder(his_vectors, lengths)
            i_vectors = self.i_embeddings(i_ids)
            prediction = (his_vector[:, None, :] * i_vectors).sum(-1)
            out_dict = {'prediction': prediction}
        return out_dict

    def loss(self, out_dict):
        if self.stage == 1:
            loss_fct = nn.BCELoss(reduction='none')
            mip_dis, mip_mask = out_dict['mip_dis'], out_dict['mip_mask']
            mip_loss = loss_fct(mip_dis, torch.ones_like(mip_dis, dtype=torch.float32))
            mip_loss = torch.sum(mip_loss * mip_mask.flatten())
            sp_dis = out_dict['sp_dis']
            sp_loss = torch.sum(loss_fct(sp_dis, torch.ones_like(sp_dis, dtype=torch.float32)))
            loss = self.mip_weight * mip_loss + self.sp_weight * sp_loss
        else:
            loss = super().loss(out_dict)
        return loss

    class Dataset(SequentialModel.Dataset):
        def __init__(self, model, corpus, phase):
            super().__init__(model, corpus, phase)
            self.pre_train = self.model.stage == 1 and self.phase == 'train'
            if self.pre_train:
                self.long_seq = list()
                item_seq, seq_len = list(), list()
                for seq in self.corpus.user_his.values():
                    instance = [x[0] for x in seq]
                    self.long_seq.extend(instance)
                    for i in range((len(instance) - 1) // self.model.max_his + 1):
                        start = i * self.model.max_his
                        end = (i + 1) * self.model.max_his
                        trunc_instance = instance[start: end]
                        item_seq.append(trunc_instance)
                        seq_len.append(len(trunc_instance))
                self.data = {'item_seq': item_seq, 'seq_len': seq_len}

        def actions_before_epoch(self):
            if self.model.stage != 1:
                super().actions_before_epoch()

        def _neg_sample(self, item_set):
            item = np.random.randint(1, self.corpus.n_items)
            while item in item_set:
                item = np.random.randint(1, self.corpus.n_items)
            return item

        def _get_mask_seq(self, seq):
            mask_token = self.model.item_num  # 0 is reserved for padding
            # MIP
            mask_seq, pos_item, neg_item = seq.copy(), seq.copy(), seq.copy()
            for idx, item in enumerate(seq):
                prob = np.random.random()
                if prob < self.model.mask_ratio:
                    mask_seq[idx] = mask_token
                    neg_item[idx] = self._neg_sample(seq)
            # SP
            if len(seq) < 2:
                mask_seg_seq, pos_seg, neg_seg = seq.copy(), seq.copy(), seq.copy()
            else:
                sample_len = np.random.randint(1, len(seq) // 2 + 1)
                start_id = np.random.randint(0, len(seq) - sample_len)
                neg_start_id = np.random.randint(0, len(self.long_seq) - sample_len)
                pos_segment = seq[start_id:start_id + sample_len]
                neg_segment = self.long_seq[neg_start_id:neg_start_id + sample_len]
                mask_seg_seq = seq[:start_id] + [mask_token] * sample_len + seq[start_id + sample_len:]
                pos_seg = [mask_token] * start_id + pos_segment + [mask_token] * (len(seq) - (start_id + sample_len))
                neg_seg = [mask_token] * start_id + neg_segment + [mask_token] * (len(seq) - (start_id + sample_len))
            return mask_seq, pos_item, neg_item, mask_seg_seq, pos_seg, neg_seg

        def _get_feed_dict(self, index):
            if self.pre_train:
                item_seq = self.data['item_seq'][index]
                mask_seq, pos_item, neg_item, mask_seg_seq, pos_seg, neg_seg = self._get_mask_seq(item_seq)
                feed_dict = {
                    'mask_seq': np.array(mask_seq),
                    'pos_item': np.array(pos_item),
                    'neg_item': np.array(neg_item),
                    'mask_seg_seq': np.array(mask_seg_seq),
                    'pos_seg': np.array(pos_seg),
                    'neg_seg': np.array(neg_seg),
                    'seq_len': self.data['seq_len'][index]
                }
            else:
                feed_dict = super()._get_feed_dict(index)
            return feed_dict


""" Encoder Layer """
class BERT4RecEncoder(nn.Module):
    def __init__(self, emb_size, max_his, num_layers=2, num_heads=2, dropout=0.2):
        super().__init__()
        self.p_embeddings = nn.Embedding(max_his + 1, emb_size)
        self.transformer_block = nn.ModuleList([
            layers.TransformerLayer(d_model=emb_size, d_ff=emb_size, n_heads=num_heads)
            for _ in range(num_layers)
        ])
        self.layer_norm = nn.LayerNorm(emb_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, seq, lengths):
        batch_size, seq_len = seq.size(0), seq.size(1)
        len_range = torch.from_numpy(np.arange(seq_len)).to(seq.device)
        valid_mask = len_range[None, :] < lengths[:, None]

        # Position embedding
        position = len_range[None, :] * valid_mask.long()
        pos_vectors = self.p_embeddings(position)
        seq = seq + pos_vectors
        seq = self.dropout(self.layer_norm(seq))

        # Self-attention
        attn_mask = valid_mask.view(batch_size, 1, 1, seq_len)
        for block in self.transformer_block:
            seq = block(seq, attn_mask)
        seq = seq * valid_mask[:, :, None].float()

        his_vector = seq[torch.arange(batch_size), lengths - 1]
        return his_vector
