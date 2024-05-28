# -*- coding: UTF-8 -*-
# @Author : Jiayu Li 
# @Email  : jy-li20@mails.tsinghua.edu.cn

""" WideDeep
Reference:
  Wide {\&} Deep Learning for Recommender Systems, Cheng et al. 2016. The 1st workshop on deep learning for recommender systems.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from models.BaseContextModel import ContextModel, ContextCTRModel
from models.context.FM import FMBase
from utils.layers import MLP_Block

class WideDeepBase(FMBase):
	@staticmethod
	def parse_model_args_WD(parser):
		parser.add_argument('--emb_size', type=int, default=64,
							help='Size of embedding vectors.')
		parser.add_argument('--layers', type=str, default='[64]',
							help="Size of each layer.")
		return parser

	def _define_init(self, args, corpus):
		self._define_init_params(args,corpus)
		self.layers = eval(args.layers)
		self._define_params_WD()
		self.apply(self.init_weights)

	def _define_params_WD(self):
		self._define_params_FM()
		pre_size = len(self.context_features) * self.vec_size
		# deep layers
		self.deep_layers = MLP_Block(pre_size, self.layers, hidden_activations="ReLU",
							   batch_norm=False, dropout_rates=self.dropout, output_dim=1)
	
	def forward(self, feed_dict):
		deep_vectors, wide_prediction = self._get_embeddings_FM(feed_dict)
		deep_vector = deep_vectors.flatten(start_dim=-2)
		deep_prediction = self.deep_layers(deep_vector).squeeze(dim=-1)
		predictions = deep_prediction + wide_prediction
		return {'prediction':predictions}
		
class WideDeepCTR(ContextCTRModel, WideDeepBase):
	reader, runner = 'ContextReader', 'CTRRunner'
	extra_log_args = ['emb_size','layers','loss_n']
	@staticmethod
	def parse_model_args(parser):
		parser = WideDeepBase.parse_model_args_WD(parser)
		return ContextModel.parse_model_args(parser)
    
	def __init__(self, args, corpus):
		ContextCTRModel.__init__(self, args, corpus)
		self._define_init(args,corpus)

	def forward(self, feed_dict):
		out_dict = WideDeepBase.forward(self, feed_dict)
		out_dict['prediction'] = out_dict['prediction'].view(-1).sigmoid()
		out_dict['label'] = feed_dict['label'].view(-1)
		return out_dict

class WideDeepTopK(ContextModel,WideDeepBase):
	reader, runner = 'ContextReader','BaseRunner'
	extra_log_args = ['emb_size','layers','loss_n']

	@staticmethod
	def parse_model_args(parser):
		parser = WideDeepBase.parse_model_args_WD(parser)
		return ContextModel.parse_model_args(parser)

	def __init__(self, args, corpus):
		ContextModel.__init__(self, args, corpus)
		self._define_init(args, corpus)

	def forward(self, feed_dict):
		return WideDeepBase.forward(self, feed_dict)
