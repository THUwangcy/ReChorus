# -*- coding: UTF-8 -*-

import logging
import numpy as np
import pandas as pd
import os
import sys

from helpers.ImpressionReader import ImpressionReader
from helpers.ContextReader import ContextReader
from helpers.BaseReader import BaseReader
from utils import utils

class ImpressionContextReader(ImpressionReader, ContextReader):
	"""
	Impression-Context Reader reads impression data and add context information to it. 
	"""

	@staticmethod
	def parse_data_args(parser):
		parser.add_argument('--include_item_features',type=int, default=0,
								help='Whether include item context features.')
		parser.add_argument('--include_user_features',type=int, default=0,
								help='Whether include user context features.')
		parser.add_argument('--include_context_features',type=int, default=0,
								help='Whether include dynamic context features.')
		return BaseReader.parse_data_args(parser)

	def __init__(self, args):
		self.sep = args.sep
		self.prefix = args.path
		self.dataset = args.dataset
		self._read_data()

		self.train_clicked_set = dict()  # store the clicked item set of each user in training set
		self.residual_clicked_set = dict()  # store the residual clicked item set of each user
		for key in ['train', 'dev', 'test']:
			df = self.data_df[key]
			for uid, iid in zip(df['user_id'], df['item_id']):
				if uid not in self.train_clicked_set:
					self.train_clicked_set[uid] = set()
					self.residual_clicked_set[uid] = set()
				if key == 'train':
					self.train_clicked_set[uid].add(iid)
				else:
					self.residual_clicked_set[uid].add(iid)
		self.include_item_features = args.include_item_features
		self.include_user_features = args.include_user_features
		self.include_context_features = args.include_context_features
		self._load_ui_metadata()
		self._collect_context()
		self._append_impression_info()
