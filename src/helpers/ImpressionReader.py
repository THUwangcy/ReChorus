# -*- coding: UTF-8 -*-

import logging
import numpy as np
import pandas as pd
import os
import sys

from helpers.BaseReader import BaseReader
from utils import utils

class ImpressionReader(BaseReader):
	"""
	Impression Reader reads impression data. In each impression there are pre-defined unfixed number of positive items and negative items
	"""
	@staticmethod
	def parse_data_args(parser):
		parser.add_argument('--impression_idkey', type=str, default='time',
                            help='The key for impression identification, [time, impression_id]')
		return BaseReader.parse_data_args(parser)
	
	def __init__(self, args):
		self.impression_idkey = args.impression_idkey
		super().__init__(args)
		self._append_impression_info()

	def _read_data(self):
		logging.info('Reading data from \"{}\", dataset = \"{}\" '.format(self.prefix, self.dataset))
		self.data_df = dict()
		for key in ['train', 'dev', 'test']:
			self.data_df[key] = pd.read_csv(os.path.join(self.prefix, self.dataset, key + '.csv'), sep=self.sep).reset_index(drop=True).sort_values(by = ['user_id',self.impression_idkey])
			self.data_df[key] = utils.eval_list_columns(self.data_df[key])
		logging.info('Counting dataset statistics...')
		if self.impression_idkey == 'time':
			key_columns = ['user_id', 'item_id', 'time']
		else:
			key_columns = ['user_id', 'item_id', 'time', self.impression_idkey]
		if 'label' in self.data_df['train'].columns:
			key_columns.append('label')
		else:
			raise KeyError('Impression data must have binary labels')
		self.all_df = pd.concat([self.data_df[key][key_columns] for key in ['train', 'dev', 'test']])
		self.n_users, self.n_items = self.all_df['user_id'].max() + 1, self.all_df['item_id'].max() + 1
		# In impression data, negative item lists can have unseen items (i.e., items without click)
		logging.info('Update impression data -- "# user": {}, "# item": {}, "# entry": {}'.format(
			self.n_users - 1, self.n_items - 1, len(self.all_df)))
		if 'label' in key_columns:
			positive_num = (self.all_df.label==1).sum()
			logging.info('"# positive interaction": {} ({:.1f}%)'.format(
				positive_num, positive_num/self.all_df.shape[0]*100))

	def _append_impression_info(self): # -> NoReturn:
		"""
		Merge all positive items of a request based on the timestamp/impression_idkey, and get column 'pos_items' for self.data_df
		Add impression info to data_df: neg_num, pos_num
		"""
		logging.info('Merging positive items by timestamp/impression_idkey...')
		# train,val,test
		mask = {'train':[],'dev':[],'test':[]}
		for key in self.data_df.keys():
			df=self.data_df[key].copy()
			df['last_user_id'] = df['user_id'].shift(1)
			df['last_'+self.impression_idkey] = df[self.impression_idkey].shift(1)

			positive_items, negative_items = [], []
			current_pos, current_neg = set(), set()
			for uid, last_uid, ipid, last_ipid, iid, label in \
					df[['user_id','last_user_id',self.impression_idkey,'last_'+self.impression_idkey,'item_id','label']].to_numpy():
				if uid == last_uid and ipid == last_ipid:
					positive_items.append([])
					negative_items.append([])
					mask[key].append(0)
				else:
					if len(current_pos):
						positive_items.append(list(current_pos))
						negative_items.append(list(current_neg))
						mask[key].append(1)
					else:
						if len(current_neg):#impression with only neg items are dropped
							positive_items.append([])
							negative_items.append([])
							mask[key].append(0)
					current_pos, current_neg = set(), set()
				if label:
					current_pos = current_pos.union(set([iid]))
				else:
					current_neg = current_neg.union(set([iid]))
			# last session
			if len(current_pos):
				positive_items.append(list(current_pos))
				negative_items.append(list(current_neg))
				mask[key].append(1)
			else:
				if len(current_neg):#impression with only neg items are dropped
					positive_items.append([])
					negative_items.append([])
					mask[key].append(0)
			self.data_df[key]['pos_items'] = positive_items
			self.data_df[key]['neg_items'] = negative_items
			self.data_df[key]=self.data_df[key][np.array(mask[key])==1]

		logging.info('Appending neg_num & pos_num...')

		neg_num_sum, pos_num_sum = 0,0
		for key in ['train', 'dev', 'test']:
			df = self.data_df[key]
			neg_num = list()
			pos_num = list()
			for neg_items in df['neg_items']:
				if 0 in neg_items:
					neg_num.append(neg_items.index(0))
				else:
					neg_num.append(len(neg_items))
			self.data_df[key]['neg_num']=neg_num
			for pos_items in df['pos_items']:
				if 0 in pos_items:
					pos_num.append(pos_items.index(0))
				else:
					pos_num.append(len(pos_items))
			self.data_df[key]['pos_num']=pos_num
			self.data_df[key] = self.data_df[key].loc[self.data_df[key].neg_num>0].reset_index(drop=True) # Retain sessions with negative data only
			neg_num_sum += sum(neg_num)
			pos_num_sum += sum(pos_num)
		neg_num_avg = neg_num_sum / sum([self.data_df[key].shape[0] for key in self.data_df])
		pos_num_avg = pos_num_sum / sum([self.data_df[key].shape[0] for key in self.data_df])
		
		logging.info('train, dev, test request num: '+str(len(self.data_df['train']))+' '+str(len(self.data_df['dev']))+' '+str(len(self.data_df['test'])))
		logging.info("Average positive items / impression = %.3f, negative items / impression = %.3f"%(
			pos_num_avg,neg_num_avg))