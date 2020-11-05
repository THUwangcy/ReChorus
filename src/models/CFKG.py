# -*- coding: UTF-8 -*-

import torch
import numpy as np
import pandas as pd

from utils import utils
from models.BaseModel import BaseModel


class CFKG(BaseModel):
    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--emb_size', type=int, default=64,
                            help='Size of embedding vectors.')
        parser.add_argument('--margin', type=float, default=0,
                            help='Margin in hinge loss.')
        return BaseModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        self.emb_size = args.emb_size
        self.margin = args.margin
        self.user_num = corpus.n_users
        self.relation_num = corpus.n_relations
        super().__init__(args, corpus)

    def _define_params(self):
        self.e_embeddings = torch.nn.Embedding(self.user_num + self.item_num, self.emb_size)
        # ↑ user and item embeddings, user first
        self.r_embeddings = torch.nn.Embedding(self.relation_num, self.emb_size)
        # ↑ relation embedding: 0-buy, 1-complement, 2-substitute
        self.loss_function = torch.nn.MarginRankingLoss(margin=self.margin)

    def forward(self, feed_dict):
        self.check_list = []
        head_ids = feed_dict['head_id']          # [batch_size, -1]
        tail_ids = feed_dict['tail_id']          # [batch_size, -1]
        relation_ids = feed_dict['relation_id']  # [batch_size, -1]

        head_vectors = self.e_embeddings(head_ids)
        tail_vectors = self.e_embeddings(tail_ids)
        relation_vectors = self.r_embeddings(relation_ids)

        prediction = -((head_vectors + relation_vectors - tail_vectors)**2).sum(-1)
        return prediction.view(feed_dict['batch_size'], -1)

    def loss(self, predictions):
        batch_size = predictions.shape[0]
        pos_pred, neg_pred = predictions[:, :2].flatten(), predictions[:, 2:].flatten()
        target = torch.from_numpy(np.ones(batch_size * 2)).to(self.device)
        loss = self.loss_function(pos_pred, neg_pred, target)
        return loss

    class Dataset(BaseModel.Dataset):
        def _prepare(self):
            if self.phase == 'train':
                interaction_df = pd.DataFrame({
                    'head': self.data['user_id'],
                    'tail': self.data['item_id'],
                    'relation': np.zeros_like(self.data['user_id'])
                })
                self.data = utils.df_to_dict(pd.concat((self.corpus.relation_df, interaction_df), axis=0))
                self.neg_heads = np.zeros(len(self), dtype=int)
                self.neg_tails = np.zeros(len(self), dtype=int)
            super()._prepare()

        def _get_feed_dict(self, index):
            if self.phase == 'train':
                head, tail = self.data['head'][index], self.data['tail'][index]
                relation = self.data['relation'][index]
                head_id = np.array([head, head, head, self.neg_heads[index]])
                tail_id = np.array([tail, tail, self.neg_tails[index], tail])
                relation_id = np.array([relation] * 4)
                if relation > 0:  # head is an item
                    head_id = head_id + self.corpus.n_users
            else:
                target_item = self.data['item_id'][index]
                neg_items = self.neg_items[index]
                tail_id = np.concatenate([[target_item], neg_items])
                head_id = self.data['user_id'][index] * np.ones_like(tail_id)
                relation_id = np.zeros_like(tail_id)
            tail_id += self.corpus.n_users  # tail must be an item

            feed_dict = {'head_id': head_id, 'tail_id': tail_id, 'relation_id': relation_id}
            return feed_dict

        def negative_sampling(self):
            for i in range(len(self)):
                head, tail, relation = self.data['head'][i], self.data['tail'][i], self.data['relation'][i]
                self.neg_tails[i] = np.random.randint(1, self.corpus.n_items)
                if relation == 0:
                    self.neg_heads[i] = np.random.randint(1, self.corpus.n_users)
                    while self.neg_tails[i] in self.corpus.user_clicked_set[head]:
                        self.neg_tails[i] = np.random.randint(1, self.corpus.n_items)
                    while tail in self.corpus.user_clicked_set[self.neg_heads[i]]:
                        self.neg_heads[i] = np.random.randint(1, self.corpus.n_users)
                else:
                    self.neg_heads[i] = np.random.randint(1, self.corpus.n_items)
                    while (head, relation, self.neg_tails[i]) in self.corpus.triplet_set:
                        self.neg_tails[i] = np.random.randint(1, self.corpus.n_items)
                    while (self.neg_heads[i], relation, tail) in self.corpus.triplet_set:
                        self.neg_heads[i] = np.random.randint(1, self.corpus.n_items)
