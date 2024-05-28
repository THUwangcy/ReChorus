# -*- coding: UTF-8 -*-
'''
Jiayu Li 2023.05.20
'''

import logging
import pandas as pd
import os
import sys

from helpers.ContextReader import ContextReader

class ContextSeqReader(ContextReader):
	def __init__(self, args):
		super().__init__(args)
		self._append_his_info()

	def _append_his_info(self):
		"""
		Similar to SeqReader, but add situation context to each history interaction.
		self.user_his: store user history sequence [(i1,t1, {situation 1}), (i1,t2, {situation 2}), ...]
		"""
		logging.info('Appending history info with history context...')
		data_dfs = dict()
		for key in ['train','dev','test']:
			data_dfs[key] = self.data_df[key].copy()
			data_dfs[key]['phase'] = key
		sort_df = pd.concat([data_dfs[phase][['user_id','item_id','time','phase']+self.situation_feature_names] 
					   for phase in ['train','dev','test']]).sort_values(by=['time', 'user_id'], kind='mergesort')
		position = list()
		self.user_his = dict()  # store the already seen sequence of each user
		situation_features = sort_df[self.situation_feature_names].to_numpy()
		for idx, (uid, iid, t) in enumerate(zip(sort_df['user_id'], sort_df['item_id'], sort_df['time'])):
			if uid not in self.user_his:
				self.user_his[uid] = list()
			position.append(len(self.user_his[uid]))
			self.user_his[uid].append((iid, t, situation_features[idx]))
		sort_df['position'] = position
		for key in ['train', 'dev', 'test']:
			self.data_df[key] = pd.merge(
				left=self.data_df[key], right=sort_df.drop(columns=['phase']+self.situation_feature_names),
				how='left', on=['user_id', 'item_id', 'time'])
		del sort_df