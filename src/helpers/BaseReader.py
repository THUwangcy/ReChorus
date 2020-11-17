# -*- coding: UTF-8 -*-

import os
import pickle
import argparse
import logging
import numpy as np
import pandas as pd
from typing import NoReturn

from utils import utils


class BaseReader(object):
    @staticmethod
    def parse_data_args(parser):
        parser.add_argument('--path', type=str, default='../data/',
                            help='Input data dir.')
        parser.add_argument('--dataset', type=str, default='Grocery_and_Gourmet_Food',
                            help='Choose a dataset.')
        parser.add_argument('--sep', type=str, default='\t',
                            help='sep of csv file.')
        parser.add_argument('--history_max', type=int, default=20,
                            help='Maximum length of history.')
        return parser

    def __init__(self, args):
        self.sep = args.sep
        self.prefix = args.path
        self.dataset = args.dataset
        self.history_max = args.history_max

        self._read_data()
        self._append_his_info()

    def _read_data(self) -> NoReturn:
        logging.info('Reading data from \"{}\", dataset = \"{}\" '.format(self.prefix, self.dataset))
        self.data_df = dict()
        for key in ['train', 'dev', 'test']:
            self.data_df[key] = pd.read_csv(os.path.join(self.prefix, self.dataset, key + '.csv'), sep=self.sep)
            self.data_df[key] = utils.eval_list_columns(self.data_df[key])

        logging.info('Counting dataset statistics...')
        self.all_df = pd.concat([df[['user_id', 'item_id', 'time']] for df in self.data_df.values()])
        self.n_users, self.n_items = self.all_df['user_id'].max() + 1, self.all_df['item_id'].max() + 1
        for key in ['dev', 'test']:
            neg_items = np.array(self.data_df[key]['neg_items'].tolist())
            assert (neg_items >= self.n_items).sum() == 0  # assert negative items don't include unseen ones
        logging.info('"# user": {}, "# item": {}, "# entry": {}'.format(self.n_users, self.n_items, len(self.all_df)))

    def _append_his_info(self) -> NoReturn:
        """
        Add history info to data_df: item_his, time_his, his_length
        ! Need data_df to be sorted by time in ascending order
        :return:
        """
        logging.info('Appending history info...')
        user_his_dict = dict()  # store the already seen sequence of each user
        for key in ['train', 'dev', 'test']:
            df = self.data_df[key]
            i_history, t_history = [], []
            for uid, iid, t in zip(df['user_id'], df['item_id'], df['time']):
                if uid not in user_his_dict:
                    user_his_dict[uid] = []
                i_history.append([x[0] for x in user_his_dict[uid]])
                t_history.append([x[1] for x in user_his_dict[uid]])
                user_his_dict[uid].append((iid, t))
            df['item_his'] = i_history
            df['time_his'] = t_history
            if self.history_max > 0:
                df['item_his'] = df['item_his'].apply(lambda x: x[-self.history_max:])
                df['time_his'] = df['time_his'].apply(lambda x: x[-self.history_max:])
            df['his_length'] = df['item_his'].apply(lambda x: len(x))

        self.user_clicked_set = dict()
        for uid in user_his_dict:
            self.user_clicked_set[uid] = set([x[0] for x in user_his_dict[uid]])


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser = BaseReader.parse_data_args(parser)
    args, extras = parser.parse_known_args()

    args.path = '../../data/'
    corpus = BaseReader(args)

    corpus_path = os.path.join(args.path, args.dataset, 'BaseReader.pkl')
    logging.info('Save corpus to {}'.format(corpus_path))
    pickle.dump(corpus, open(corpus_path, 'wb'))
