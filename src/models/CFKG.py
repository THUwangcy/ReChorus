# -*- coding: UTF-8 -*-

import torch
import numpy as np
import pandas as pd

from utils import utils
from models.BaseModel import BaseModel


class CFKG(BaseModel):
    @staticmethod
    def parse_model_args(parser, model_name='CFKG'):
        parser.add_argument('--emb_size', type=int, default=64,
                            help='Size of embedding vectors.')
        parser.add_argument('--margin', type=float, default=0,
                            help='Margin in hinge loss.')
        return BaseModel.parse_model_args(parser, model_name)

    def __init__(self, args, corpus):
        self.emb_size = args.emb_size
        self.margin = args.margin
        self.user_num = corpus.n_users
        self.item_num = corpus.n_items
        self.relation_num = corpus.n_relations
        BaseModel.__init__(self, model_path=args.model_path)

    def _define_params(self):
        # user and item embedding
        self.e_embeddings = torch.nn.Embedding(self.user_num + self.item_num, self.emb_size)
        # relation embedding: 0-buy, 1-complement, 2-substitute
        self.r_embeddings = torch.nn.Embedding(self.relation_num, self.emb_size)
        self.embeddings = ['e_embeddings', 'r_embeddings']

        self.loss_function = torch.nn.MarginRankingLoss(margin=self.margin)

    def forward(self, feed_dict):
        self.check_list, self.embedding_l2 = [], []
        head_ids = feed_dict['head_id']
        tail_ids = feed_dict['tail_id']
        relation_ids = feed_dict['relation_id']

        head_vectors = self.e_embeddings(head_ids)
        tail_vectors = self.e_embeddings(tail_ids)
        relation_vectors = self.r_embeddings(relation_ids)
        self.embedding_l2.extend([head_vectors, tail_vectors, relation_vectors])

        prediction = -((head_vectors + relation_vectors - tail_vectors)**2).sum(dim=-1)

        out_dict = {'prediction': prediction, 'check': self.check_list}
        return out_dict

    def loss(self, feed_dict, predictions):
        real_batch_size = feed_dict['batch_size']
        pos_pred, neg_pred = predictions[:real_batch_size * 2], predictions[real_batch_size * 2:]
        loss = self.loss_function(pos_pred, neg_pred, utils.numpy_to_torch(np.ones(real_batch_size * 2)))
        return loss.double()

    def get_feed_dict(self, corpus, data, batch_start, batch_size, phase):
        batch_end = min(len(data), batch_start + batch_size)
        real_batch_size = batch_end - batch_start

        if phase == 'train':
            head_ids = data['head'][batch_start: batch_start + real_batch_size].values
            tail_ids = data['tail'][batch_start: batch_start + real_batch_size].values
            relation_ids = data['relation'][batch_start: batch_start + real_batch_size].values
            neg_heads, neg_tails = self.sample_negative_triplet(
                corpus, real_batch_size, head_ids, tail_ids, relation_ids)
            head_ids = head_ids + (relation_ids > 0).astype(int) * self.user_num
            neg_heads = neg_heads + (relation_ids > 0).astype(int) * self.user_num
            head_ids = np.concatenate((head_ids, head_ids, head_ids, neg_heads))
            tail_ids = np.concatenate((tail_ids, tail_ids, neg_tails, tail_ids))
            relation_ids = np.tile(relation_ids, 4)
        else:
            head_ids = data['user_id'][batch_start: batch_start + real_batch_size].values
            item_ids = data['item_id'][batch_start: batch_start + real_batch_size].values
            neg_items = data['neg_items'][batch_start: batch_start + real_batch_size].tolist()
            tail_ids = np.concatenate([np.expand_dims(item_ids, -1), neg_items], axis=1).reshape(-1)
            n_candidates = len(tail_ids) // len(head_ids)
            head_ids = np.expand_dims(head_ids, -1).repeat(n_candidates, axis=1).reshape(-1)
            relation_ids = np.zeros_like(head_ids)
        tail_ids = tail_ids + self.user_num

        feed_dict = {
            'head_id': utils.numpy_to_torch(head_ids),
            'tail_id': utils.numpy_to_torch(tail_ids),
            'relation_id': utils.numpy_to_torch(relation_ids),
            'batch_size': real_batch_size
        }
        return feed_dict

    def prepare_batches(self, corpus, data, batch_size, phase):
        if phase == 'train':
            interaction_data = pd.DataFrame({
                'head': data['user_id'].values,
                'tail': data['item_id'].values,
                'relation': np.zeros_like(data['user_id'].values)
            })
            data = pd.concat((corpus.relation_df, interaction_data), axis=0)
            data = data.sample(frac=1).reset_index(drop=True)
        return BaseModel.prepare_batches(self, corpus, data, batch_size, phase)

    def sample_negative_triplet(self, corpus, real_batch_size, head_ids, tail_ids, relation_ids):
        neg_tails = np.random.randint(1, self.item_num, size=real_batch_size)
        neg_heads = np.zeros(real_batch_size, dtype=int)
        for i in range(real_batch_size):
            if relation_ids[i] == 0:
                while neg_tails[i] in corpus.user_clicked_set[head_ids[i]]:
                    neg_tails[i] = np.random.randint(1, self.item_num)
                neg_heads[i] = np.random.randint(1, self.user_num)
                while tail_ids[i] in corpus.user_clicked_set[neg_heads[i]]:
                    neg_heads[i] = np.random.randint(1, self.user_num)
            else:
                while (head_ids[i], relation_ids[i], neg_tails[i]) in corpus.triplet_set:
                    neg_tails[i] = np.random.randint(1, self.item_num)
                neg_heads[i] = np.random.randint(1, self.item_num)
                while (neg_heads[i], relation_ids[i], tail_ids[i]) in corpus.triplet_set:
                    neg_heads[i] = np.random.randint(1, self.item_num)
        return neg_heads, neg_tails
