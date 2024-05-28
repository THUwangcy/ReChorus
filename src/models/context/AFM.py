# -*- coding: UTF-8 -*-
# @Author : Zhiyu He
# @Email  : hezy22@mails.tsinghua.edu.cn

""" AFM
Reference:
	'Attentional Factorization Machines: Learning the Weight of Feature Interactions via Attention Networks', Xiao et al, 2017. Arxiv.
Implementation reference: AFM and RecBole
	https://github.com/hexiangnan/attentional_factorization_machine
	https://github.com/RUCAIBox/RecBole/blob/master/recbole/model/context_aware_recommender/afm.py
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from models.BaseContextModel import ContextCTRModel, ContextModel
from models.context.FM import FMBase
from utils.layers import AttLayer # move here?

class AFMBase(FMBase):
	@staticmethod
	def parse_model_args_AFM(parser):
		parser.add_argument('--emb_size', type=int, default=64,
							help='Size of embedding vectors.')
		parser.add_argument('--attention_size', type=int, default=64,
							help='Size of attention embedding vectors.')
		parser.add_argument('--reg_weight',type=float, default=2.0,
                      		help='Regularization weight for attention layer weights.')
		return parser	

	def _define_init_afm(self, args, corpus):
		self.vec_size = args.emb_size
		self.attention_size = args.attention_size
		self.reg_weight = args.reg_weight
		self._define_params_AFM()
		self.apply(self.init_weights)

	def _define_params_AFM(self):
		self._define_params_FM() # basic embedding initialization from FM
		self.dropout_layer = nn.Dropout(p=self.dropout)
		self.attlayer = AttLayer(self.vec_size, self.attention_size)
		self.p = torch.nn.Parameter(torch.randn(self.vec_size),requires_grad=True)

	def build_cross(self, feat_emb):
		row = []
		col = []
		for i in range(len(self.feature_max)-1):
			for j in range(i+1, len(self.feature_max)):
				row.append(i)
				col.append(j)
		p = feat_emb[:,:,row]
		q = feat_emb[:,:,col]
		return p, q

	def afm_layer(self, infeature):
		"""Reference:
			RecBole - https://github.com/RUCAIBox/RecBole/blob/master/recbole/model/context_aware_recommender/afm.py
		"""
		p, q = self.build_cross(infeature)
		pair_wise_inter = torch.mul(p,q) # batch_size * num_items * num_pairs * emb_dim

		att_signal = self.attlayer(pair_wise_inter).unsqueeze(dim=-1)
		att_inter = torch.mul(
			att_signal, pair_wise_inter
		)  # [batch_size, num_items, num_pairs, emb_dim]
		att_pooling = torch.sum(att_inter, dim=-2)  # [batch_size, num_items, emb_dim]
		att_pooling = self.dropout_layer(att_pooling)  # [batch_size, num_items, emb_dim]

		att_pooling = torch.mul(att_pooling, self.p)  # [batch_size, num_items, emb_dim]
		att_pooling = torch.sum(att_pooling, dim=-1, keepdim=True)  # [batch_size, num_items, 1]
		return att_pooling

	def forward(self, feed_dict):
		fm_vectors, linear_value = self._get_embeddings_FM(feed_dict)

		afm_vectors = self.afm_layer(fm_vectors)
		predictions = linear_value + afm_vectors.squeeze(dim=-1)

		return {'prediction':predictions}
	
class AFMCTR(ContextCTRModel, AFMBase): # CTR mode
	reader, runner = 'ContextReader', 'CTRRunner'
	extra_log_args = ['emb_size', 'attention_size', 'loss_n']

	@staticmethod
	def parse_model_args(parser):
		parser = AFMBase.parse_model_args_AFM(parser)
		return ContextModel.parse_model_args(parser)

	def __init__(self, args, corpus):
		ContextCTRModel.__init__(self, args, corpus)
		self._define_init_afm(args,corpus)
	
	def forward(self, feed_dict):
		out_dict = AFMBase.forward(self, feed_dict)
		out_dict['prediction'] = out_dict['prediction'].view(-1).sigmoid()
		out_dict['label'] = feed_dict['label'].view(-1)
		return out_dict
	
	def loss(self, out_dict: dict):
		l2_loss = self.reg_weight * torch.norm(self.attlayer.w.weight, p=2)
		loss = ContextCTRModel.loss(self, out_dict)
		return loss + l2_loss

class AFMTopK(ContextModel,AFMBase): # Top-k Ranking mode
	reader, runner = 'ContextReader', 'BaseRunner'
	extra_log_args = ['emb_size', 'attention_size', 'loss_n']

	@staticmethod
	def parse_model_args(parser):
		parser = AFMBase.parse_model_args_AFM(parser)
		return ContextModel.parse_model_args(parser)

	def __init__(self, args, corpus):
		ContextModel.__init__(self, args, corpus)
		self._define_init_afm(args,corpus)

	def forward(self, feed_dict):
		return AFMBase.forward(self, feed_dict)

	def loss(self, out_dict: dict):
		l2_loss = self.reg_weight * torch.norm(self.attlayer.w.weight, p=2)
		loss = ContextModel.loss(self, out_dict)
		return loss + l2_loss
