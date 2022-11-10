# -*- coding: UTF-8 -*-

import os
import pickle
import argparse
import logging
import numpy as np
import pandas as pd

from utils import utils


class GuideReader(object):
    @staticmethod
    def parse_data_args(parser):
        parser.add_argument('--path', type=str, default='../data/',
                            help='Input data dir.')
        parser.add_argument('--dataset', type=str, default='Grocery_and_Gourmet_Food',
                            help='Choose a dataset.')
        parser.add_argument('--sep', type=str, default='\t',
                            help='sep of csv file.')
        return parser

    def __init__(self, args):
        self.sep = args.sep
        self.prefix = args.path
        self.dataset = args.dataset
        self._read_data()

        self.train_clicked_set = dict()  # store the clicked item set of each user in training set
        self.residual_clicked_set = dict()  # store the residual clicked item set of each user
        for key in ['train', 'dev', 'test']:
            df = self.data_df[key]
            for uid, iid in zip(df['user_id'], df['item_id']):
                if uid not in self.train_clicked_set:
                    self.train_clicked_set[uid] = set()
                    self.residual_clicked_set[uid] = set()
                if key == 'train':
                    self.train_clicked_set[uid].add(iid)
                else:
                    self.residual_clicked_set[uid].add(iid)

        candidate_dataset = ['ml-500k', 'cmcc', 'QK-article-1M']
        if self.dataset in candidate_dataset:
            self._build_sets()

    def _read_data(self):
        logging.info('Reading data from \"{}\", dataset = \"{}\" '.format(self.prefix, self.dataset))
        self.data_df = dict()
        for key in ['train', 'dev', 'test']:
            self.data_df[key] = pd.read_csv(os.path.join(self.prefix, self.dataset, key + '.csv'), sep=self.sep)
            self.data_df[key] = utils.eval_list_columns(self.data_df[key])

        logging.info('Counting dataset statistics...')
        self.all_df = pd.concat([self.data_df[key][['user_id', 'item_id', 'time']] for key in ['train', 'dev', 'test']])
        self.n_users, self.n_items = self.all_df['user_id'].max() + 1, self.all_df['item_id'].max() + 1
        for key in ['dev', 'test']:
            if 'neg_items' in self.data_df[key]:
                neg_items = np.array(self.data_df[key]['neg_items'].tolist())
                assert (neg_items >= self.n_items).sum() == 0  # assert negative items don't include unseen ones
        logging.info('"# user": {}, "# item": {}, "# entry": {}'.format(
            self.n_users - 1, self.n_items - 1, len(self.all_df)))

    def _build_sets(self):
        item_meta_path = os.path.join(self.prefix, self.dataset, 'item_meta.csv')
        self.item_meta_df = pd.read_csv(item_meta_path, sep=self.sep)
        self.item_meta_df['item_score3'] = self.item_meta_df['item_score3'].apply(lambda x: 9 if x > 9 else x)

        self.item2quality = dict(zip(self.item_meta_df['item_id'], self.item_meta_df['item_score3']))

        self.item_set = set(self.item_meta_df['item_id'])  # 商品全集

        self.HQI_set = set(self.item_meta_df[self.item_meta_df['i_quality'] > 0]['item_id'])
        # print("imdb_pivot: {}".format(db_pivot))
        print("#HQI: {}, #HQI frac: {}".format(len(self.HQI_set), len(self.HQI_set)/len(self.item_set)))
