# -*- coding: UTF-8 -*-

import torch
import logging
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from typing import List

from utils import utils
from models.BaseModel import *

def get_context_feature(feed_dict, index, corpus, data):
	"""
	Get context features for the feed_dict, including user, item, and situation context
 	"""
	for c in corpus.user_feature_names:
		feed_dict[c] = corpus.user_features[feed_dict['user_id']][c]
	for c in corpus.situation_feature_names:
		feed_dict[c] = data[c][index]
	for c in corpus.item_feature_names:
		if type(feed_dict['item_id']) in [int, np.int32, np.int64]: # for a single item
			feed_dict[c] = corpus.item_features[feed_dict['item_id']][c]
		else: # for item list
			feed_dict[c] = np.array([corpus.item_features[iid][c] for iid in feed_dict['item_id']])
	return feed_dict

class ContextModel(GeneralModel):
	# context model for top-k recommendation tasks
	reader = 'ContextReader'
	
	@staticmethod
	def parse_model_args(parser):
		parser.add_argument('--loss_n',type=str,default='BPR',
							help='Type of loss functions.')
		return GeneralModel.parse_model_args(parser)

	def __init__(self, args, corpus):
		super().__init__(args, corpus)
		self.loss_n = args.loss_n
		self.context_features = corpus.user_feature_names + corpus.item_feature_names + corpus.situation_feature_names\
					+ ['user_id','item_id']
		self.feature_max = corpus.feature_max
	
	def loss(self, out_dict: dict):
		"""
		utilize BPR loss (same as general models) or BCE loss (same as CTR tasks)
		"""
		if self.loss_n == 'BPR':
			loss = super().loss(out_dict)
		elif self.loss_n == 'BCE':
			predictions = out_dict['prediction'].sigmoid()
			pos_pred, neg_pred = predictions[:, 0], predictions[:, 1:]
			loss = - (pos_pred.log() + (1-neg_pred).log().sum(dim=1)).mean()
		else:
			raise ValueError('Undefined loss function: {}'.format(self.loss_n))
		if torch.isnan(loss) or torch.isinf(loss):
			print('Error!')
		return loss
	
	class Dataset(GeneralModel.Dataset):
		def __init__(self, model, corpus, phase):
			super().__init__(model, corpus, phase)

		def _get_feed_dict(self, index):
			feed_dict = super()._get_feed_dict(index)
			feed_dict = get_context_feature(feed_dict, index, self.corpus, self.data)
			return feed_dict


class ContextCTRModel(CTRModel):
	# context model for CTR prediction tasks
	reader = 'ContextReader'

	def __init__(self, args, corpus):
		super().__init__(args, corpus)
		self.context_features = corpus.user_feature_names + corpus.item_feature_names + corpus.situation_feature_names\
					+ ['user_id','item_id']
		self.feature_max = corpus.feature_max

	class Dataset(CTRModel.Dataset):
		def _get_feed_dict(self, index):
			feed_dict = super()._get_feed_dict(index)
			feed_dict = get_context_feature(feed_dict, index, self.corpus, self.data)
			return feed_dict

class ContextSeqModel(ContextModel):
	reader='ContextSeqReader'
	
	@staticmethod
	def parse_model_args(parser):
		parser.add_argument('--history_max', type=int, default=20,
							help='Maximum length of history.')
		parser.add_argument('--add_historical_situations',type=int,default=0,
					  		help='Whether to add historical situation context as sequence.')
		return ContextModel.parse_model_args(parser)
	
	def __init__(self, args, corpus):
		super().__init__(args, corpus)
		self.history_max = args.history_max
		self.add_historical_situations = args.add_historical_situations

	class Dataset(SequentialModel.Dataset):
		def __init__(self, model, corpus, phase):
			super().__init__(model, corpus, phase)
		
		def _get_feed_dict(self, index):
			# get item features, user features, and context features separately
			feed_dict = super()._get_feed_dict(index)
			pos = self.data['position'][index]
			user_seq = self.corpus.user_his[feed_dict['user_id']][:pos]
			if self.model.history_max > 0:
				user_seq = user_seq[-self.model.history_max:]
			feed_dict = get_context_feature(feed_dict, index, self.corpus, self.data)
			for c in self.corpus.item_feature_names: # get historical item context features
				feed_dict['history_'+c] = np.array([self.corpus.item_features[iid][c] for iid in feed_dict['history_items']])
			if self.model.add_historical_situations: # get historical situation context features
				for idx,c in enumerate(self.corpus.situation_feature_names):
					feed_dict['history_'+c] = np.array([inter[-1][idx] for inter in user_seq])
			feed_dict['history_item_id'] = feed_dict['history_items']
			feed_dict.pop('history_items')
			return feed_dict

class ContextSeqCTRModel(ContextCTRModel):
	reader = 'ContextSeqReader'
	
	@staticmethod
	def parse_model_args(parser):
		parser.add_argument('--history_max', type=int, default=20,
							help='Maximum length of history.')
		parser.add_argument('--add_historical_situations',type=int,default=0,
					  		help='Whether to add historical situation context as sequence.')
		return ContextCTRModel.parse_model_args(parser)
	
	def __init__(self, args, corpus):
		super().__init__(args, corpus)
		self.history_max = args.history_max
		self.add_historical_situations = args.add_historical_situations

	class Dataset(ContextCTRModel.Dataset):
		def __init__(self, model, corpus, phase):
			super().__init__(model, corpus, phase)
			idx_select = np.array(self.data['position']) > 0  # history length must be non-zero
			for key in self.data:
				self.data[key] = np.array(self.data[key])[idx_select]
		
		def _get_feed_dict(self, index):
			feed_dict = super()._get_feed_dict(index)
			pos = self.data['position'][index]
			user_seq = self.corpus.user_his[feed_dict['user_id']][:pos]
			if self.model.history_max > 0:
				user_seq = user_seq[-self.model.history_max:]
			# feed_dict = get_context_feature(feed_dict, index, self.corpus, self.data)
			feed_dict['history_items'] = np.array([x[0] for x in user_seq])
			feed_dict['history_times'] = np.array([x[1] for x in user_seq])
			feed_dict['lengths'] = len(feed_dict['history_items'])
			for c in self.corpus.item_feature_names: # get historical item context features
				feed_dict['history_'+c] = np.array([self.corpus.item_features[iid][c] for iid in feed_dict['history_items']])
			if self.model.add_historical_situations: # get historical situation context features
				for idx,c in enumerate(self.corpus.situation_feature_names):
					feed_dict['history_'+c] = np.array([inter[-1][idx] for inter in user_seq])
			feed_dict['history_item_id'] = feed_dict['history_items']
			feed_dict.pop('history_items')
			return feed_dict
