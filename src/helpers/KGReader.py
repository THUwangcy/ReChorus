# -*- coding: UTF-8 -*-

import os
import time
import pickle
import argparse
import logging
import numpy as np
import pandas as pd
from typing import NoReturn
from helpers.BaseReader import BaseReader


class KGReader(BaseReader):
    @staticmethod
    def parse_data_args(parser):
        return BaseReader.parse_data_args(parser)

    def __init__(self, args):
        super().__init__(args)
        self._construct_kg()

    def _construct_kg(self) -> NoReturn:
        logging.info('Constructing relation triplets...')

        self.triplet_set = set()
        heads, relations, tails = [], [], []
        relation_types = [r for r in self.item_meta_df.columns if r.startswith('r_')]
        for idx in range(len(self.item_meta_df)):
            head_item = self.item_meta_df['item_id'][idx]
            for r_idx, r in enumerate(relation_types):
                for tail_item in self.item_meta_df[r][idx]:
                    heads.append(head_item)
                    relations.append(r_idx + 1)
                    tails.append(tail_item)
                    self.triplet_set.add((head_item, r_idx + 1, tail_item))

        self.relation_df = pd.DataFrame()
        self.relation_df['head'] = heads
        self.relation_df['relation'] = relations
        self.relation_df['tail'] = tails
        self.n_relations = self.relation_df['relation'].max() + 1
        logging.info('"# relation": {}, "# triplet": {}'.format(self.n_relations, len(self.relation_df)))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser = BaseReader.parse_data_args(parser)
    args, extras = parser.parse_known_args()

    args.path = '../../data/'
    corpus = BaseReader(args)

    corpus_path = os.path.join(args.path, args.dataset, 'KGReader.pkl')
    logging.info('Save corpus to {}'.format(corpus_path))
    pickle.dump(corpus, open(corpus_path, 'wb'))
