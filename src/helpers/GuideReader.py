# -*- coding: UTF-8 -*-

import os
import pandas as pd

from helpers.SeqReader import SeqReader


class GuideReader(SeqReader):
    def __init__(self, args):
        super().__init__(args)
        candidate_dataset = ['ml-500k', 'cmcc', 'QK-article-1M']
        if self.dataset in candidate_dataset:
            self._build_sets()

    def _build_sets(self):
        item_meta_path = os.path.join(self.prefix, self.dataset, 'item_meta.csv')
        self.item_meta_df = pd.read_csv(item_meta_path, sep=self.sep)

        if self.dataset == 'QK-article-1M':
            self.item_meta_df['item_score3'] = self.item_meta_df['item_score3'].apply(lambda x: 9 if x > 9 else x)

        self.item2quality = dict(zip(self.item_meta_df['item_id'], self.item_meta_df['i_quality']))
        self.quality_level = int(self.item_meta_df['i_quality'].max()) + 1

        self.item_set = set(self.item_meta_df['item_id'])  # 商品全集
        self.HQI_set = set(self.item_meta_df[self.item_meta_df['i_quality'] > 0]['item_id'])

        print("#HQI: {}, #HQI frac: {}".format(len(self.HQI_set), len(self.HQI_set)/len(self.item_set)))
