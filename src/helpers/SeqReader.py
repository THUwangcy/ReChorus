# -*- coding: UTF-8 -*-

import logging
import pandas as pd

from helpers.BaseReader import BaseReader


class SeqReader(BaseReader):
    def __init__(self, args):
        super().__init__(args)
        self._append_his_info()

    def _append_his_info(self):
        """
        self.user_his: store user history sequence [(i1,t1), (i1,t2), ...]
        add the 'position' of each interaction in user_his to data_df
        """
        logging.info('Appending history info...')
        sort_df = self.all_df.sort_values(by=['time', 'user_id'], kind='mergesort')
        position = list()
        self.user_his = dict()  # store the already seen sequence of each user
        for uid, iid, t in zip(sort_df['user_id'], sort_df['item_id'], sort_df['time']):
            if uid not in self.user_his:
                self.user_his[uid] = list()
            position.append(len(self.user_his[uid]))
            self.user_his[uid].append((iid, t))
        sort_df['position'] = position
        for key in ['train', 'dev', 'test']:
            self.data_df[key] = pd.merge(
                left=self.data_df[key], right=sort_df, how='left',
                on=['user_id', 'item_id', 'time'])
        del sort_df