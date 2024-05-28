# -*- coding: UTF-8 -*-
# @Author : Zhiyu He
# @Email  : hezy22@mails.tsinghua.edu.cn

""" DCN v2
Reference:
	'DCN v2: Improved deep & cross network and practical lessons for web-scale learning to rank systems.', Wang et al, WWW2021.
Implementation reference: RecBole
	https://github.com/RUCAIBox/RecBole/blob/master/recbole/model/context_aware_recommender/dcnv2.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as fn
import numpy as np
import pandas as pd

from models.BaseContextModel import ContextCTRModel, ContextModel
from models.context.DCN import DCNBase
from utils.layers import MLP_Block

class DCNv2Base(DCNBase):
	@staticmethod
	def parse_model_args_DCNv2Base(parser):
		parser = DCNBase.parse_model_args_DCN(parser)
		parser.add_argument('--mixed',type=int, default=1,
                      		help="Wether user mixed cross network or not.")
		parser.add_argument('--structure',type=str, default='parallel',
                      		help="cross network and DNN is 'parallel' or 'stacked'")
		parser.add_argument('--low_rank',type=int, default=64,
                      		help="Size for the low-rank architecture when mixed==1.")
		parser.add_argument('--expert_num', type=int, default=2,
                      		help="Number of experts to calculate in each cross layer when mixed==1.")
		return parser

	def _define_init(self, args, corpus):
		self._define_init_params(args,corpus)
		self.mixed = args.mixed
		self.structure = args.structure
		self.expert_num = args.expert_num
		self.low_rank = args.low_rank

		self._define_params_DCNv2()
		self.apply(self.init_weights)

	def _define_params_DCNv2(self):
		# embedding
		self.context_embedding = nn.ModuleDict()
		for f in self.context_features:
			self.context_embedding[f] = nn.Embedding(self.feature_max[f],self.vec_size) if f.endswith('_c') or f.endswith('_id') else\
					nn.Linear(1,self.vec_size,bias=False)

		pre_size = len(self.feature_max) * self.vec_size
		# cross layers
		if self.mixed:
			self.cross_layer_u = nn.ParameterList(nn.Parameter(torch.randn(self.expert_num, pre_size, self.low_rank))
                            for l in range(self.cross_layer_num)) # U: (pre_size, low_rank)
			self.cross_layer_v = nn.ParameterList(nn.Parameter(torch.randn(self.expert_num, pre_size, self.low_rank))
                            for l in range(self.cross_layer_num)) # V: (pre_size, low_rank)
			self.cross_layer_c = nn.ParameterList(nn.Parameter(torch.randn(self.expert_num, self.low_rank, self.low_rank))
                            for l in range(self.cross_layer_num)) # V: (pre_size, low_rank)
			self.gating = nn.ModuleList(nn.Linear(pre_size, 1) for l in range(self.expert_num))
		else:
			self.cross_layer_w2 = nn.ParameterList(nn.Parameter(torch.randn(pre_size, pre_size))
                			for l in range(self.cross_layer_num)) # W: (pre_size, pre_size)
		self.bias = nn.ParameterList(nn.Parameter(torch.zeros(pre_size, 1))for l in range(self.cross_layer_num))
		self.tanh = nn.Tanh()

		# deep layers
		self.deep_layers = MLP_Block(pre_size, self.layers, hidden_activations="ReLU",
							   batch_norm=True,norm_before_activation=True,
          						dropout_rates=self.dropout, output_dim=None)
		# output layer
		if self.structure == "parallel":
			self.predict_layer = nn.Linear(len(self.feature_max) * self.vec_size + self.layers[-1], 1)
		if self.structure == "stacked":
			self.predict_layer = nn.Linear(self.layers[-1], 1)
		
	def cross_net_2(self, x_0):
		# x_0: batch size * item num * embedding size
		'''
		math:: x_{l+1} = x_0 * {W_l · x_l + b_l} + x_l
        '''
		batch_size, item_num, output_emb = x_0.shape
		x_0 = x_0.view(-1,  output_emb)
		x_0 = x_0.unsqueeze(dim=2)
		x_l = x_0 # x0: (batch_size * num_item, context_num * emb_size, 1)
		for layer in range(self.cross_layer_num): # self.cross_layer_w2[layer]: (context_num * emb_size, context_num * emb_size)
			xl_w = torch.matmul(self.cross_layer_w2[layer], x_l) # wl_w: (batch_size * num_item, context_num * emb_size, 1)
			xl_w = xl_w + self.bias[layer] # self.bias[layer]: (context_num * emb_size, 1)
			xl_dot = torch.mul(x_0, xl_w) # wl_dot: (batch_size * num_item, context_num * emb_size, 1)
			x_l = xl_dot + x_l
		x_l = x_l.view(batch_size, item_num, -1)
		return x_l

	def cross_net_mix(self, x_0):
		"""Reference: RecBole - https://github.com/RUCAIBox/RecBole/blob/master/recbole/model/context_aware_recommender/dcnv2.py
		add MoE and nonlinear transformation in low-rank space
        .. math::
            x_{l+1} = \sum_{i=1}^K G_i(x_l)E_i(x_l)+x_l
        .. math::
            E_i(x_l) = x_0 \odot (U_l^i \dot g(C_l^i \dot g(V_L^{iT} x_l)) + b_l)

        :math:`E_i` and :math:`G_i` represents the expert and gatings respectively,
        :math:`U_l`, :math:`C_l`, :math:`V_l` stand for low-rank decomposition of weight matrix,
        :math:`g` is the nonlinear activation function.

        Args:
            x_0(torch.Tensor): Embedding vectors of all features, input of cross network.

        Returns:
            torch.Tensor:output of mixed cross network, [batch_size, num_feature_field * embedding_size]
        """
		# x_0: batch size * item num * embedding size
		batch_size, item_num, output_emb = x_0.shape
		x_0 = x_0.view(-1,  output_emb) # x0: (batch_size * num_item, context_num * emb_size)
		x_0 = x_0.unsqueeze(dim=2)
		x_l = x_0 # x0: (batch_size * num_item, context_num * emb_size, 1)
		for layer in range(self.cross_layer_num):
			expert_output_list = []
			gating_output_list = []
			for expert in range(self.expert_num):
                # compute gating output
				gating_output_list.append(self.gating[expert](x_l.squeeze(dim=2)))  # (batch_size, 1)
                # project to low-rank subspace
				xl_v = torch.matmul(self.cross_layer_v[layer][expert].T, x_l)  # (batch_size, low_rank, 1)
                # nonlinear activation in subspace
				xl_c = nn.Tanh()(xl_v)
				xl_c = torch.matmul(self.cross_layer_c[layer][expert], xl_c)  # (batch_size, low_rank, 1)
				xl_c = nn.Tanh()(xl_c)
                # project back feature space
				xl_u = torch.matmul(self.cross_layer_u[layer][expert], xl_c)  # (batch_size, in_feature_num, 1)
                # dot with x_0
				xl_dot = xl_u + self.bias[layer]
				xl_dot = torch.mul(x_0, xl_dot)
				expert_output_list.append(xl_dot.squeeze(dim=2))  # (batch_size, in_feature_num)

			expert_output = torch.stack(expert_output_list, dim=2)  # (batch_size, in_feature_num, expert_num)
			gating_output = torch.stack(gating_output_list, dim=1)  # (batch_size, expert_num, 1)
			moe_output = torch.matmul(expert_output, nn.Softmax(dim=1)(gating_output))  # (batch_size, in_feature_num, 1)
			x_l = x_l + moe_output

		x_l = x_l.view(batch_size, item_num, -1)
		return x_l

	def forward(self, feed_dict):
		item_ids = feed_dict['item_id']
		batch_size, item_num = item_ids.shape

		context_vectors = [self.context_embedding[f](feed_dict[f]) if f.endswith('_c') or f.endswith('_id') 
						  else self.context_embedding[f](feed_dict[f].float().unsqueeze(-1))
						  for f in self.context_features]
		context_vectors = torch.stack([v if len(v.shape)==3 else v.unsqueeze(dim=-2).repeat(1, item_num, 1) 
							for v in context_vectors], dim=-2) # batch size * item num * feature num * feature dim
		context_emb = context_vectors.flatten(start_dim=-2)

		if self.mixed:
			cross_output = self.cross_net_mix(context_emb)
		else:
			cross_output = self.cross_net_2(context_emb)
		batch_size, item_num, output_emb = cross_output.shape
		if self.structure == 'parallel':
			deep_output = context_emb.view(-1,output_emb) # (batch_size * num_item, context_num * emb_size)
			deep_output = self.deep_layers(deep_output).view(batch_size, item_num, self.layers[-1]) # (batch_size, num_item, context_num * emb_size)
			output = self.predict_layer(torch.cat([cross_output, deep_output],dim=-1)) 
		elif self.structure == 'stacked':
			deep_output = cross_output.view(-1,output_emb)
			deep_output = self.deep_layers(deep_output).view(batch_size, item_num, self.layers[-1]) # (batch_size, num_item, context_num * emb_size)
			output = self.predict_layer(deep_output)

		predictions = output.squeeze(dim=-1)
		return {'prediction':predictions}
class DCNv2CTR(ContextCTRModel, DCNv2Base):
	reader, runner = 'ContextReader', 'CTRRunner'
	extra_log_args = ['emb_size','loss_n','cross_layer_num']
	
	@staticmethod
	def parse_model_args(parser):
		parser = DCNv2Base.parse_model_args_DCNv2Base(parser)
		return ContextCTRModel.parse_model_args(parser)

	def __init__(self, args, corpus):
		ContextCTRModel.__init__(self, args, corpus)
		self._define_init(args,corpus)

	def forward(self, feed_dict):
		out_dict = DCNv2Base.forward(self, feed_dict)
		out_dict['prediction'] = out_dict['prediction'].view(-1).sigmoid()
		out_dict['label'] = feed_dict['label'].view(-1)
		return out_dict

	def loss(self, out_dict: dict):
		loss = ContextCTRModel.loss(self, out_dict)
		if not self.mixed:
			l2_loss = self.reg_weight * DCNv2Base.l2_reg(self, self.cross_layer_w2)
			return loss + l2_loss
		else:
			return loss

class DCNv2TopK(ContextModel, DCNv2Base):
	reader, runner = 'ContextReader', 'BaseRunner'
	extra_log_args = ['emb_size','loss_n','cross_layer_num']
	
	@staticmethod
	def parse_model_args(parser):
		parser = DCNv2Base.parse_model_args_DCNv2Base(parser)
		return ContextModel.parse_model_args(parser)

	def __init__(self, args, corpus):
		ContextModel.__init__(self, args, corpus)
		self._define_init(args,corpus)

	def forward(self, feed_dict):
		return DCNv2Base.forward(self, feed_dict)
	
	def loss(self, out_dict: dict):
		loss = ContextModel.loss(self, out_dict)
		if not self.mixed:
			l2_loss = self.reg_weight * DCNv2Base.l2_reg(self, self.cross_layer_w2)
			return loss + l2_loss
		else:
			return loss