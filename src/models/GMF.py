# -*- coding: UTF-8 -*-

import torch
import numpy as np

from utils import utils
from models.BaseModel import BaseModel


class GMF(BaseModel):
    @staticmethod
    def parse_model_args(parser, model_name='GMF'):
        parser.add_argument('--layers', type=str, default='[64]',
                            help="Size of each layer.")
        parser.add_argument('--num_neg', type=int, default=4,
                            help="Number of negative samples.")
        return BaseModel.parse_model_args(parser, model_name)

    def __init__(self, args, corpus):
        self.layers = eval(args.layers)
        self.emb_size = self.layers[0]
        self.num_neg = args.num_neg
        self.dropout = args.dropout
        self.user_num = corpus.n_users
        self.item_num = corpus.n_items
        BaseModel.__init__(self, model_path=args.model_path)

    def _define_params(self):
        self.u_embeddings = torch.nn.Embedding(self.user_num, self.emb_size)
        self.i_embeddings = torch.nn.Embedding(self.item_num, self.emb_size)
        for i, layer_size in enumerate(self.layers[1:]):
            setattr(self, 'layer_%d' % i, torch.nn.Linear(self.layers[i], layer_size))
        self.prediction = torch.nn.Linear(self.layers[-1], 1, bias=False)
        self.embeddings = ['u_embeddings', 'i_embeddings']

    def forward(self, feed_dict):
        self.check_list, self.embedding_l2 = [], []
        u_ids = feed_dict['user_id']  # [real_batch_size]
        i_ids = feed_dict['item_id']
        batch_size = feed_dict['batch_size']

        i_ids = i_ids.view(batch_size, -1)
        u_vectors = self.u_embeddings(u_ids)
        i_vectors = self.i_embeddings(i_ids)
        self.embedding_l2.extend([u_vectors, i_vectors])

        mf_vector = u_vectors[:, None, :] * i_vectors
        for i in range(len(self.layers) - 1):
            mf_vector = getattr(self, 'layer_%d' % i)(mf_vector).relu()
            mf_vector = torch.nn.Dropout(p=self.dropout)(mf_vector)

        prediction = self.prediction(mf_vector).flatten()

        out_dict = {'prediction': prediction, 'check': self.check_list}
        return out_dict

    def loss(self, feed_dict, predictions):
        loss = torch.nn.BCEWithLogitsLoss()(predictions, feed_dict['label'])
        return loss

    def get_feed_dict(self, corpus, data, batch_start, batch_size, phase):
        batch_end = min(len(data), batch_start + batch_size)
        real_batch_size = batch_end - batch_start
        user_ids = data['user_id'][batch_start: batch_start + real_batch_size].values
        item_ids = data['item_id'][batch_start: batch_start + real_batch_size].values
        labels = np.ones_like(item_ids, dtype=np.float64)

        if phase == 'train':
            neg_items = np.random.randint(1, self.item_num, size=(real_batch_size, self.num_neg))
            for i in range(real_batch_size):
                for j in range(self.num_neg):
                    while neg_items[i][j] in corpus.user_clicked_set[user_ids[i]]:
                        neg_items[i][j] = np.random.randint(1, self.item_num)
        else:
            neg_items = data['neg_items'][batch_start: batch_start + real_batch_size].tolist()

        neg_labels = np.zeros_like(neg_items, dtype=np.float64)
        labels = np.concatenate((np.expand_dims(labels, -1), neg_labels), axis=1).reshape(-1)
        item_ids = np.concatenate([np.expand_dims(item_ids, -1), neg_items], axis=1).reshape(-1)

        feed_dict = {
            'user_id': utils.numpy_to_torch(user_ids),  # [real_batch_size]
            'item_id': utils.numpy_to_torch(item_ids),
            'label': utils.numpy_to_torch(labels),
            'batch_size': real_batch_size
        }
        return feed_dict
