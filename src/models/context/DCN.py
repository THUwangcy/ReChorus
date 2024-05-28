# -*- coding: UTF-8 -*-
# @Author : Zhiyu He
# @Email  : hezy22@mails.tsinghua.edu.cn

""" DCN
Reference:
	'Deep & Cross Network for Ad Click Predictions', Wang et al, KDD2017.
Implementation reference: RecBole
	https://github.com/RUCAIBox/RecBole/blob/master/recbole/model/context_aware_recommender/dcn.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as fn
import numpy as np
import pandas as pd

from models.BaseContextModel import ContextCTRModel, ContextModel
from utils.layers import MLP_Block

class DCNBase(object):
	@staticmethod
	def parse_model_args_DCN(parser):
		parser.add_argument('--emb_size', type=int, default=64,
							help='Size of embedding vectors.')
		parser.add_argument('--layers', type=str, default='[64]',
							help="Size of each deep layer.")
		parser.add_argument('--cross_layer_num',type=int,default=6,
							help="Number of cross layers.")
		parser.add_argument('--reg_weight',type=float, default=2.0,
                      		help="Regularization weight for cross-layer weights. In DCNv2, it is only used for mixed version")
		return parser

	def _define_init(self, args, corpus):
		self._define_init_params(args,corpus)
		self._define_params_DCN()
		self.apply(self.init_weights)

	def _define_init_params(self, args, corpus):
		self.vec_size = args.emb_size
		self.reg_weight = args.reg_weight
		self.layers = eval(args.layers)
		self.cross_layer_num = args.cross_layer_num

	def _define_params_DCN(self):
		# embedding
		self.context_embedding = nn.ModuleDict()
		for f in self.context_features:
			self.context_embedding[f] = nn.Embedding(self.feature_max[f],self.vec_size) if f.endswith('_c') or f.endswith('_id') else\
					nn.Linear(1,self.vec_size,bias=False)
		pre_size = len(self.feature_max) * self.vec_size
		# cross layers
		self.cross_layer_w = nn.ParameterList(nn.Parameter(torch.randn(pre_size),requires_grad=True) 
							for l in range(self.cross_layer_num))
		self.cross_layer_b = nn.ParameterList(nn.Parameter(torch.tensor([0.01]*pre_size),requires_grad=True)
							for l in range(self.cross_layer_num))
		# deep layers
		self.deep_layers = MLP_Block(pre_size, self.layers, hidden_activations="ReLU",
							   batch_norm=True,norm_before_activation=True,
          						dropout_rates=self.dropout, output_dim=None)
		# output layer
		self.predict_layer = nn.Linear(len(self.feature_max) * self.vec_size + self.layers[-1], 1)

	def cross_net(self, x_0):
		# x_0: batch size * item num * embedding size
		'''
		math:: x_{l+1} = x_0 · w_l * x_l^T + b_l + x_l
		'''
		x_l = x_0
		for layer in range(self.cross_layer_num):
			xl_w = torch.tensordot(x_l, self.cross_layer_w[layer], dims=([-1],[0]))
			xl_dot = x_0 * xl_w.unsqueeze(2)
			x_l = xl_dot + self.cross_layer_b[layer] + x_l
		return x_l

	def forward(self, feed_dict):
		item_ids = feed_dict['item_id']
		batch_size, item_num = item_ids.shape
		# embedding
		context_vectors = [self.context_embedding[f](feed_dict[f]) if f.endswith('_c') or f.endswith('_id') 
						  else self.context_embedding[f](feed_dict[f].float().unsqueeze(-1))
						  for f in self.context_features]
		context_vectors = torch.stack([v if len(v.shape)==3 else v.unsqueeze(dim=-2).repeat(1, item_num, 1) 
							for v in context_vectors], dim=-2) # batch size * item num * feature num * feature dim
		context_emb = context_vectors.flatten(start_dim=-2)

		# cross net
		cross_output = self.cross_net(context_emb)
		batch_size, item_num, output_emb = cross_output.shape
		deep_output = context_emb.view(-1,output_emb)
		deep_output = self.deep_layers(deep_output)
		deep_output = deep_output.view(batch_size, item_num, self.layers[-1])

		# output		
		output = self.predict_layer(torch.cat([cross_output, deep_output],dim=-1))
		predictions = output.squeeze(dim=-1)
		return {'prediction':predictions}

	def l2_reg(self, parameters):
		"""
		Reference: 
		RecBole - https://github.com/RUCAIBox/RecBole/blob/master/recbole/model/loss.py
		RegLoss, L2 regularization on model parameters
		"""
		reg_loss = None
		for W in parameters:
			if reg_loss is None:
				reg_loss = W.norm(2)
			else:
				reg_loss = reg_loss + W.norm(2)
		return reg_loss

class DCNCTR(ContextCTRModel, DCNBase):
	reader, runner = 'ContextReader', 'CTRRunner'
	extra_log_args = ['emb_size','loss_n','cross_layer_num']
	
	@staticmethod
	def parse_model_args(parser):
		parser = DCNBase.parse_model_args_DCN(parser)
		return ContextCTRModel.parse_model_args(parser)

	def __init__(self, args, corpus):
		ContextCTRModel.__init__(self, args, corpus)
		self._define_init(args,corpus)

	def forward(self, feed_dict):
		out_dict = DCNBase.forward(self, feed_dict)
		out_dict['prediction'] = out_dict['prediction'].view(-1).sigmoid()
		out_dict['label'] = feed_dict['label'].view(-1)
		return out_dict

	def loss(self, out_dict: dict):
		l2_loss = self.reg_weight * DCNBase.l2_reg(self, self.cross_layer_w)
		loss = ContextCTRModel.loss(self, out_dict)
		return loss + l2_loss

class DCNTopK(ContextModel, DCNBase):
	reader, runner = 'ContextReader', 'BaseRunner'
	extra_log_args = ['emb_size','loss_n','cross_layer_num']
	
	@staticmethod
	def parse_model_args(parser):
		parser = DCNBase.parse_model_args_DCN(parser)
		return ContextModel.parse_model_args(parser)

	def __init__(self, args, corpus):
		ContextModel.__init__(self, args, corpus)
		self._define_init(args,corpus)

	def forward(self, feed_dict):
		return DCNBase.forward(self, feed_dict)

	def loss(self, out_dict: dict):
		l2_loss = self.reg_weight * DCNBase.l2_reg(self, self.cross_layer_w)
		loss = ContextModel.loss(self, out_dict)
		return loss + l2_loss