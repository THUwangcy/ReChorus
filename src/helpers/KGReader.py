# -*- coding: UTF-8 -*-

import os
import pickle
import argparse
import logging
import numpy as np
import pandas as pd


from helpers.SeqReader import SeqReader
from utils import utils


class KGReader(SeqReader):
    @staticmethod
    def parse_data_args(parser):
        parser.add_argument('--include_attr', type=int, default=0,
                            help='Whether include attribute-based relations.')
        return SeqReader.parse_data_args(parser)

    def __init__(self, args):
        super().__init__(args)
        self.include_attr = args.include_attr
        item_meta_path = os.path.join(self.prefix, self.dataset, 'item_meta.csv')
        self.item_meta_df = pd.read_csv(item_meta_path, sep=self.sep)
        self.item_meta_df = utils.eval_list_columns(self.item_meta_df)

        self._construct_kg()

    def _construct_kg(self):
        logging.info('Constructing relation triplets...')

        self.triplet_set = set()
        heads, relations, tails = [], [], []

        self.item_relations = [r for r in self.item_meta_df.columns if r.startswith('r_')]
        for idx in range(len(self.item_meta_df)):
            head_item = self.item_meta_df['item_id'].values[idx]
            for r_idx, r in enumerate(self.item_relations):
                for tail_item in self.item_meta_df[r].values[idx]:
                    heads.append(head_item)
                    tails.append(tail_item)
                    relations.append(r_idx + 1)  # idx 0 is reserved to be a virtual relation between items
                    self.triplet_set.add((head_item, r_idx + 1, tail_item))
        logging.info('Item-item relations:' + str(self.item_relations))

        self.attr_relations = list()
        if self.include_attr:
            self.attr_relations = [r for r in self.item_meta_df.columns if r.startswith('i_')]
            self.attr_max, self.share_attr_dict = list(), dict()
            for r_idx, attr in enumerate(self.attr_relations):
                base = self.n_items + np.sum(self.attr_max)  # base index of attribute entities
                relation_idx = len(self.item_relations) + r_idx + 1  # index of the relation type
                for item, val in zip(self.item_meta_df['item_id'], self.item_meta_df[attr]):
                    if val != 0:  # the attribute is not NaN
                        heads.append(item)
                        tails.append(int(val + base))
                        relations.append(relation_idx)
                        self.triplet_set.add((item, relation_idx, int(val + base)))
                for val, val_df in self.item_meta_df.groupby(attr):
                    self.share_attr_dict[int(val + base)] = val_df['item_id'].tolist()
                self.attr_max.append(self.item_meta_df[attr].max() + 1)
            logging.info('Attribute-based relations:' + str(self.attr_relations))

        self.relations = self.item_relations + self.attr_relations
        self.relation_df = pd.DataFrame()
        self.relation_df['head'] = heads
        self.relation_df['relation'] = relations
        self.relation_df['tail'] = tails
        self.n_relations = len(self.relations) + 1
        self.n_entities = pd.concat((self.relation_df['head'], self.relation_df['tail'])).max() + 1
        logging.info('"# relation": {}, "# triplet": {}'.format(self.n_relations, len(self.relation_df)))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser = KGReader.parse_data_args(parser)
    args, extras = parser.parse_known_args()

    args.path = '../../data/'
    corpus = KGReader(args)

    corpus_path = os.path.join(args.path, args.dataset, 'KGReader.pkl')
    logging.info('Save corpus to {}'.format(corpus_path))
    pickle.dump(corpus, open(corpus_path, 'wb'))
