 # -*- coding: UTF-8 -*-
# @Author : Zhiyu He
# @Email  : hezy22@mails.tsinghua.edu.cn

"""AutoInt
Reference:
	Weiping Song et al. "AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks"
	in CIKM 2018.
Implementation reference: AutoInt and FuxiCTR
	https://github.com/shichence/AutoInt/blob/master/model.py
	https://github.com/reczoo/FuxiCTR/blob/main/model_zoo/AutoInt/src/AutoInt.py
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from models.BaseContextModel import ContextCTRModel, ContextModel
from models.context.FM import FMBase
from utils.layers import MultiHeadAttention, MLP_Block

class AutoIntBase(FMBase):
	@staticmethod
	def parse_model_args_AutoInt(parser):
		parser.add_argument('--emb_size', type=int, default=64,
							help='Size of embedding vectors.')
		parser.add_argument('--attention_size', type=int, default=32,
							help='Size of attention hidden space.')
		parser.add_argument('--num_heads', type=int, default=1,
							help='Number of attention heads.')
		parser.add_argument('--num_layers', type=int, default=1,
							help='Number of self-attention layers.') # for attention layer
		parser.add_argument('--layers', type=str, default='[64]',
							help="Size of each layer.") # for deep layers
		return parser 
	
	def _define_init(self, args, corpus):
		self.vec_size = args.emb_size
		self.layers = eval(args.layers)
  
		self.num_heads = args.num_heads
		self.num_layers = args.num_layers
		self.attention_size= args.attention_size

		self._define_params_AutoInt()
		self.apply(self.init_weights)

	def _define_params_AutoInt(self):
		self._define_params_FM()
		# Attention
		att_input = self.vec_size
		autoint_attentions = []
		residual_embeddings = []
		for _ in range(self.num_layers):
			autoint_attentions.append(
				MultiHeadAttention(d_model=att_input, n_heads=self.num_heads, kq_same=False, bias=False,
									attention_d=self.attention_size))
			residual_embeddings.append(nn.Linear(att_input, self.attention_size))
			att_input = self.attention_size
		self.autoint_attentions = nn.ModuleList(autoint_attentions)
		self.residual_embeddings = nn.ModuleList(residual_embeddings)
		# Deep
		pre_size = len(self.feature_max) * self.attention_size
		self.deep_layers = MLP_Block(pre_size, self.layers, hidden_activations="ReLU",
							   dropout_rates=self.dropout, output_dim=1)

	def forward(self, feed_dict):
		# FM
		autoint_all_embeddings, linear_value = self._get_embeddings_FM(feed_dict)
		# Attention + Residual
		for autoint_attention, residual_embedding in zip(self.autoint_attentions, self.residual_embeddings):
			attention = autoint_attention(autoint_all_embeddings, autoint_all_embeddings, autoint_all_embeddings)
			residual = residual_embedding(autoint_all_embeddings)
			autoint_all_embeddings = (attention + residual).relu()
		# Deep
		deep_vectors = autoint_all_embeddings.flatten(start_dim=-2)
		deep_vectors = self.deep_layers(deep_vectors)
		predictions = linear_value + deep_vectors.squeeze(-1)
		return {'prediction':predictions}

class AutoIntCTR(ContextCTRModel, AutoIntBase):
	reader, runner = 'ContextReader', 'CTRRunner'
	extra_log_args = ['emb_size','layers','num_layers','num_heads','loss_n']
	
	@staticmethod
	def parse_model_args(parser):
		parser = AutoIntBase.parse_model_args_AutoInt(parser)
		return ContextCTRModel.parse_model_args(parser)

	def __init__(self, args, corpus):
		ContextCTRModel.__init__(self, args, corpus)
		self._define_init(args,corpus)
	
	def forward(self, feed_dict):
		out_dict = AutoIntBase.forward(self, feed_dict)
		out_dict['prediction'] = out_dict['prediction'].view(-1).sigmoid()
		out_dict['label'] = feed_dict['label'].view(-1)
		return out_dict	

class AutoIntTopK(ContextModel,AutoIntBase):
	reader, runner = 'ContextReader', 'BaseRunner'
	extra_log_args = ['emb_size','layers','num_layers','num_heads','loss_n']
	
	@staticmethod
	def parse_model_args(parser):
		parser = AutoIntBase.parse_model_args_AutoInt(parser)
		return ContextModel.parse_model_args(parser)

	def __init__(self, args, corpus):
		ContextModel.__init__(self, args, corpus)
		self._define_init(args,corpus)
	
	def forward(self, feed_dict):
		return AutoIntBase.forward(self, feed_dict)
