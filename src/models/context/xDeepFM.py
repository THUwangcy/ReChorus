# -*- coding: UTF-8 -*-
# @Author : Zhiyu He
# @Email  : hezy22@mails.tsinghua.edu.cn

""" Reference:
	"xdeepfm: Combining explicit and implicit feature interactions for recommender systems". Lian et al. KDD2018.
Implementation reference: xDeeoFM and RecBole
	https://github.com/Leavingseason/xDeepFM/blob/master/exdeepfm/src/exDeepFM.py/
	https://github.com/RUCAIBox/RecBole/blob/master/recbole/model/context_aware_recommender/xdeepfm.py
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from models.BaseContextModel import ContextCTRModel, ContextModel
from models.context.FM import FMBase
from utils.layers import MLP_Block

class xDeepFMBase(FMBase):
	@staticmethod
	def parse_model_args_xDeepFM(parser):
		parser.add_argument('--emb_size', type=int, default=64, help='Size of embedding vectors.')
		parser.add_argument('--layers', type=str, default='[64]', help="Size of each layer.")
		parser.add_argument('--cin_layers',type=str,default='[8,8]', help="Size of each layer.")
		parser.add_argument('--direct', type=int, default=0,
							help="Whether utilize the output of current network for the next layer.")
		parser.add_argument('--reg_weight',type=float, default=2.0, help="The weight of regularization loss term.")
		return parser
	
	def _define_init(self, args, corpus):
		self.vec_size = args.emb_size
		self.layers = eval(args.layers)
		self.reg_weight = args.reg_weight
		
		self.direct = args.direct
		self.cin_layer_size = temp_cin_size = eval(args.cin_layers)
		# Check whether the size of the CIN layer is legal.
		if not self.direct:
			self.cin_layer_size = list(map(lambda x: int(x // 2 * 2), temp_cin_size))
			if self.cin_layer_size[:-1] != temp_cin_size[:-1]:
				self.logger.warning(
					"Layer size of CIN should be even except for the last layer when direct is True."
					"It is changed to {}".format(self.cin_layer_size)
				)
	
		self._define_params_xDeepFM()
		self.apply(self.init_weights)
	
	def _define_params_xDeepFM(self):
		# FM
		self._define_params_FM()
		# CIN
		self.conv1d_list = nn.ModuleList()
		self.field_nums = [len(self.feature_max)]
		for i, layer_size in enumerate(self.cin_layer_size):
			conv1d = nn.Conv1d(self.field_nums[-1] * self.field_nums[0], layer_size, 1)
			self.conv1d_list.append(conv1d)
			if self.direct:
				self.field_nums.append(layer_size)
			else:
				self.field_nums.append(layer_size // 2)
		if self.direct:
			self.final_len = sum(self.cin_layer_size)
		else:
			self.final_len = (
				sum(self.cin_layer_size[:-1]) // 2 + self.cin_layer_size[-1]
			)
		self.cin_linear = nn.Linear(self.final_len, 1)
		# Deep
		pre_size = len(self.feature_max) * self.vec_size
		self.deep_layers = MLP_Block(pre_size, self.layers, hidden_activations="ReLU",
							   batch_norm=False, dropout_rates=self.dropout, output_dim=1)
	
	def l2_reg(self, parameters):
		"""
		Reference: RecBole - https://github.com/RUCAIBox/RecBole/blob/master/recbole/model/loss.py
		Calculate the L2 normalization loss of parameters in a certain layer.
		Returns:
			loss(torch.FloatTensor): The L2 Loss tensor. shape of [1,]
		"""
		reg_loss = 0
		for name, parm in parameters:
			if name.endswith("weight"):
				reg_loss = reg_loss + parm.norm(2)
		return reg_loss

	def reg_loss(self):
		l2_reg_loss = self.l2_reg(self.deep_layers.named_parameters()) + self.l2_reg(self.linear_embedding.named_parameters())
		for conv1d in self.conv1d_list:
			l2_reg_loss += self.l2_reg(conv1d.named_parameters())
		return l2_reg_loss

	def compreseed_interaction_network(self, input_features, activation="nn.ReLU"):
		"""Reference:
			RecBole - https://github.com/RUCAIBox/RecBole/blob/master/recbole/model/context_aware_recommender/xdeepfm.py
		"""
		batch_size, item_num, feature_num, embedding_size = input_features.shape
		all_item_result = []
		for item_idx in range(item_num):
			hidden_nn_layers = [input_features[:,item_idx,:,:]]
			final_result = []
			for i, layer_size in enumerate(self.cin_layer_size):
				z_i = torch.einsum( "bhd,bmd->bhmd", hidden_nn_layers[-1], hidden_nn_layers[0])
				z_i = z_i.view(
					batch_size, self.field_nums[0] * self.field_nums[i], embedding_size
				)
				z_i = self.conv1d_list[i](z_i)
				# Pass the CIN intermediate result through the activation function.
				if activation.lower() == "identity" or activation == "None":
					output = z_i
				else:
					activate_func = eval(activation)()
					output = activate_func(z_i)
				# Get the output of the hidden layer.
				if self.direct:
					direct_connect = output
					next_hidden = output
				else:
					if i != len(self.cin_layer_size) - 1:
						next_hidden, direct_connect = torch.split(
							output, 2 * [layer_size // 2], 1
						)
					else:
						direct_connect = output
						next_hidden = 0
				final_result.append(direct_connect)
				hidden_nn_layers.append(next_hidden)
			result = torch.cat(final_result, dim=1)
			result = torch.sum(result, dim=-1)
			all_item_result.append(result.unsqueeze(1))
		all_item_result = torch.cat(all_item_result,dim=1)
		return result

	def forward(self, feed_dict):
		item_ids = feed_dict['item_id']
		batch_size, item_num = item_ids.shape
		# FM
		context_vectors, fm_prediction = self._get_embeddings_FM(feed_dict)
		fm_vectors = 0.5 * (context_vectors.sum(dim=-2).pow(2) - context_vectors.pow(2).sum(dim=-2)) # batch size * item num * feature dim
		fm_prediction = fm_prediction + fm_vectors.sum(dim=-1)
		# deep
		deep_vectors = self.deep_layers(context_vectors.flatten(start_dim=-2))
		deep_prediction = deep_vectors.squeeze(dim=-1)
		# CIN
		cin_output = self.compreseed_interaction_network(context_vectors)
		cin_output = self.cin_linear(cin_output).squeeze(dim=-1)

		predictions = fm_prediction + deep_prediction
		return {'prediction':predictions}

class xDeepFMCTR(ContextCTRModel, xDeepFMBase):
	reader, runner = 'ContextReader', 'CTRRunner'
	extra_log_args = ['emb_size','layers','loss_n']

	@staticmethod
	def parse_model_args(parser):
		parser = xDeepFMBase.parse_model_args_xDeepFM(parser)
		return ContextCTRModel.parse_model_args(parser)
	
	def __init__(self, args, corpus):
		ContextCTRModel.__init__(self, args, corpus)
		self._define_init(args,corpus)

	def forward(self, feed_dict):
		out_dict = xDeepFMBase.forward(self, feed_dict)
		out_dict['prediction'] = out_dict['prediction'].view(-1).sigmoid()
		out_dict['label'] = feed_dict['label'].view(-1)
		return out_dict

	def loss(self, out_dict: dict):
		l2_loss = self.reg_weight * xDeepFMBase.reg_loss(self)
		loss = ContextCTRModel.loss(self, out_dict)
		return loss + l2_loss


class xDeepFMTopK(ContextModel, xDeepFMBase):
	reader, runner = 'ContextReader', 'BaseRunner'
	extra_log_args = ['emb_size','layers','loss_n']

	@staticmethod
	def parse_model_args(parser):
		parser = xDeepFMBase.parse_model_args_xDeepFM(parser)
		return ContextModel.parse_model_args(parser)
	
	def __init__(self, args, corpus):
		ContextModel.__init__(self, args, corpus)
		self._define_init(args,corpus)

	def forward(self, feed_dict):
		return xDeepFMBase.forward(self, feed_dict)

	def loss(self, out_dict: dict):
		l2_loss = self.reg_weight * xDeepFMBase.reg_loss(self)
		loss = ContextModel.loss(self, out_dict)
		return loss + l2_loss
