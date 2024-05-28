# -*- coding: UTF-8 -*-

import logging
import numpy as np
import pandas as pd
import os
import sys

from helpers.ImpressionReader import ImpressionReader
from utils import utils

class ImpressionSeqReader(ImpressionReader):
	
	def __init__(self, args):
		super().__init__(args)
		self._append_his_info()

	def _append_his_info(self):
		"""
		self.user_his: store both positive and negative user history sequences 
  			pos: [(i1,t1), (i1,t2), ...];
     		neg: [(in1,tn1), (in2,tn2),...]
		add the 'position' of each interaction in user_his to data_df
		"""
		logging.info('Appending history info with corresponding impressions...')
		data_dfs = dict()
		for key in ['train','dev','test']:
			data_dfs[key] = self.data_df[key].copy()
			data_dfs[key]['phase'] = key
		if self.impression_idkey == 'time':
			key_columns = ['user_id', 'pos_items', 'neg_items', 'time', 'phase']
			sort_columns = ['user_id', 'time']
		else:
			key_columns = ['user_id', 'pos_items', 'neg_items', 'time', 'phase', self.impression_idkey]
			sort_columns = ['user_id', self.impression_idkey, 'time']
		sort_df = pd.concat([data_dfs[phase][key_columns] 
					   for phase in ['train','dev','test']]).sort_values(by=sort_columns, kind='mergesort')
		position = list()
		neg_position = list()
		self.user_his = dict()  # store the already seen sequence of each user
		for idx, (uid, pids, nids, t) in enumerate(zip(sort_df['user_id'], sort_df['pos_items'], sort_df['neg_items'], sort_df['time'])):
			if uid not in self.user_his:
				self.user_his[uid] = {'pos':list(),'neg':list()}
			position.append(len(self.user_his[uid]['pos']))
			neg_position.append(len(self.user_his[uid]['neg']))
			for pid in pids:
				self.user_his[uid]['pos'].append((pid, t))
			for nid in nids:
				self.user_his[uid]['neg'].append((nid, t))
		sort_df['position'] = position
		sort_df['neg_position'] = neg_position
		for key in ['train', 'dev', 'test']:
			self.data_df[key] = pd.merge(
				left=self.data_df[key], right=sort_df.drop(columns=['phase','pos_items','neg_items']),
				how='left', on=['user_id', self.impression_idkey])
		del sort_df