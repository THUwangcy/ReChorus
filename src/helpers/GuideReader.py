# -*- coding: UTF-8 -*-

import os
import pandas as pd
import numpy as np

from helpers.SeqReader import SeqReader


class GuideReader(SeqReader):
    def __init__(self, args):
        super().__init__(args)
        candidate_dataset = ['CMCC', 'QK-article-1M']
        if self.dataset in candidate_dataset:
            self._build_sets()
            self._user_his_dist()

    def _build_sets(self):
        item_meta_path = os.path.join(self.prefix, self.dataset, 'item_meta.csv')
        self.item_meta_df = pd.read_csv(item_meta_path, sep=self.sep)

        if self.dataset == 'QK-article-1M':
            self.item_meta_df['item_score3'] = self.item_meta_df['item_score3'].apply(lambda x: 9 if x > 9 else x)

        self.item2quality = dict(zip(self.item_meta_df['item_id'], self.item_meta_df['i_quality']))
        self.quality_level = int(self.item_meta_df['i_quality'].max()) + 1

        self.item_set = set(self.item_meta_df['item_id'])
        self.HQI_set = set(self.item_meta_df[self.item_meta_df['i_quality'] > 0]['item_id'])  # high-quality item set

        print("#HQI: {}, #HQI frac: {}".format(len(self.HQI_set), len(self.HQI_set)/len(self.item_set)))

    def _user_his_dist(self):
        self.p_u = dict()
        df = self.data_df['train']
        for uid, iid in zip(df['user_id'], df['item_id']):
            if uid not in self.p_u:
                self.p_u[uid] = [0] * self.quality_level
            quality = self.item2quality[iid]
            self.p_u[uid][quality] += 1
        for uid in self.p_u.keys():
            dist = np.array(self.p_u[uid])
            self.p_u[uid] = dist / dist.sum()
