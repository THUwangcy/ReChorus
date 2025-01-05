# -*- coding: UTF-8 -*-
# @Author : Jiayu Li 
# @Email  : jy-li20@mails.tsinghua.edu.cn

""" 
Reference:
	'Deep interest network for click-through rate prediction', Zhou et al., SIGKDD2018.
Implementation reference:  
	RecBole: https://github.com/RUCAIBox/RecBole
	DIN pytorch repo: https://github.com/fanoping/DIN-pytorch
"""

import torch
import torch.nn as nn
import torch.nn.functional as fn
import numpy as np
import pandas as pd

from models.BaseContextModel import ContextSeqModel, ContextSeqCTRModel
from utils.layers import MLP_Block

class DINBase(object):
	@staticmethod
	def parse_model_args_din(parser):
		parser.add_argument('--emb_size', type=int, default=64,
							help='Size of embedding vectors.')
		parser.add_argument('--att_layers', type=str, default='[64]',
							help="Size of each layer in the attention module.")
		parser.add_argument('--dnn_layers', type=str, default='[64]',
							help="Size of each layer in the MLP module.")
		return parser

	def _define_init(self, args,corpus):
		self.user_context = ['user_id']+corpus.user_feature_names
		self.item_context = ['item_id']+corpus.item_feature_names
		self.situation_context = corpus.situation_feature_names
		self.item_feature_num = len(self.item_context)
		self.user_feature_num = len(self.user_context)
		self.situation_feature_num = len(corpus.situation_feature_names) if self.add_historical_situations else 0
  
		self.vec_size = args.emb_size
		self.att_layers = eval(args.att_layers)
		self.dnn_layers = eval(args.dnn_layers)
		self._define_params_DIN()
		self.apply(self.init_weights)

	def _define_params_DIN(self):
		self.embedding_dict = nn.ModuleDict()
		for f in self.user_context+self.item_context+self.situation_context:
			self.embedding_dict[f] = nn.Embedding(self.feature_max[f],self.vec_size) if f.endswith('_c') or f.endswith('_id') else\
					nn.Linear(1,self.vec_size,bias=False)

		pre_size = 4 * (self.item_feature_num+self.situation_feature_num) * self.vec_size
		self.att_mlp_layers = MLP_Block(input_dim=pre_size,hidden_units=self.att_layers,output_dim=1,
                                  hidden_activations='Sigmoid',dropout_rates=self.dropout,batch_norm=False)

		pre_size = (2*(self.item_feature_num+self.situation_feature_num)+self.item_feature_num
              +len(self.situation_context) + self.user_feature_num) * self.vec_size
		self.dnn_mlp_layers = MLP_Block(input_dim=pre_size,hidden_units=self.dnn_layers,output_dim=1,
                                  hidden_activations='Dice',dropout_rates=self.dropout,batch_norm=True,
                                  norm_before_activation=True)

	def attention(self, queries, keys, keys_length, mask_mat,softmax_stag=False, return_seq_weight=False):
		'''Reference:
			RecBole layers: SequenceAttLayer, https://github.com/RUCAIBox/RecBole/blob/master/recbole/model/layers.py#L294
			queries: batch * (if*vecsize)
		'''
		embedding_size = queries.shape[-1]  # H
		hist_len = keys.shape[1]  # T
		queries = queries.repeat(1, hist_len)
		queries = queries.view(-1, hist_len, embedding_size)
		# MLP Layer
		input_tensor = torch.cat(
			[queries, keys, queries - keys, queries * keys], dim=-1
		)
		output = torch.transpose(self.att_mlp_layers(input_tensor), -1, -2)
		# get mask
		output = output.squeeze(1)
		mask = mask_mat.repeat(output.size(0), 1)
		mask = mask >= keys_length.unsqueeze(1)
		# mask
		if softmax_stag:
			mask_value = -np.inf
		else:
			mask_value = 0.0
		output = output.masked_fill(mask=mask, value=torch.tensor(mask_value))
		output = output.unsqueeze(1)
		output = output / (embedding_size**0.5)
		# get the weight of each user's history list about the target item
		if softmax_stag:
			output = fn.softmax(output, dim=2)  # [B, 1, T]
		if not return_seq_weight:
			output = torch.matmul(output, keys)  # [B, 1, H]
		torch.cuda.empty_cache()
		return output.squeeze(dim=1)

	def get_all_embedding(self, feed_dict, merge_all=True):
		# item embedding
		item_feats_emb = torch.stack([self.embedding_dict[f](feed_dict[f]) if f.endswith('_c') or f.endswith('_id')
					else self.embedding_dict[f](feed_dict[f].float().unsqueeze(-1))
					for f in self.item_context],dim=-2) # batch * feature num * emb size
		# historical item embedding
		history_item_emb = torch.stack([self.embedding_dict[f](feed_dict['history_'+f]) if f.endswith('_c') or f.endswith('_id')
					else self.embedding_dict[f](feed_dict['history_'+f].float().unsqueeze(-1))
					for f in self.item_context],dim=-2) # batch * feature num * emb size
		# user embedding
		user_feats_emb = torch.stack([self.embedding_dict[f](feed_dict[f]) if f.endswith('_c') or f.endswith('_id')
					else self.embedding_dict[f](feed_dict[f].float().unsqueeze(-1))
					for f in self.user_context],dim=-2) # batch * feature num * emb size
		# situation embedding
		if len(self.situation_context):
			situ_feats_emb = torch.stack([self.embedding_dict[f](feed_dict[f]) if f.endswith('_c') or f.endswith('_id')
					else self.embedding_dict[f](feed_dict[f].float().unsqueeze(-1))
					for f in self.situation_context],dim=-2) # batch * feature num * emb size
		else:
			situ_feats_emb = None
		# historical situation embedding
		if self.add_historical_situations and len(self.situation_context):
			history_situ_emb = torch.stack([self.embedding_dict[f](feed_dict['history_'+f]) if f.endswith('_c') or f.endswith('_id')
					else self.embedding_dict[f](feed_dict['history_'+f].float().unsqueeze(-1))
					for f in self.situation_context],dim=-2) # batch * feature num * emb size
			history_emb = torch.cat([history_item_emb,history_situ_emb],dim=-2).flatten(start_dim=-2)
			item_num = item_feats_emb.shape[1]
			current_emb = torch.cat([item_feats_emb, situ_feats_emb.unsqueeze(1).repeat(1,item_num,1,1)],dim=-2).flatten(start_dim=-2)
		else:
			history_emb = history_item_emb.flatten(start_dim=-2)
			current_emb = item_feats_emb.flatten(start_dim=-2)

		if merge_all:
			item_num = item_feats_emb.shape[1]
			if situ_feats_emb is not None:
				all_context = torch.cat([item_feats_emb, user_feats_emb.unsqueeze(1).repeat(1,item_num,1,1),
							situ_feats_emb.unsqueeze(1).repeat(1,item_num,1,1)],dim=-2).flatten(start_dim=-2)
			else:
				all_context = torch.cat([item_feats_emb, user_feats_emb.unsqueeze(1).repeat(1,item_num,1,1),
							],dim=-2).flatten(start_dim=-2)
					
			return history_emb, current_emb, all_context
		else:
			return history_emb, current_emb, user_feats_emb, situ_feats_emb

	def forward(self, feed_dict):
		hislens = feed_dict['lengths']
		history_emb, current_emb, all_context = self.get_all_embedding(feed_dict)
		predictions = self.att_dnn(current_emb,history_emb, all_context, hislens)
		return {'prediction':predictions}

	def att_dnn(self, current_emb, history_emb, all_context, history_lengths):
		mask_mat = (torch.arange(history_emb.shape[1]).view(1,-1)).to(self.device)
  
		batch_size, item_num, feats_emb = current_emb.shape
		_, max_len, his_emb = history_emb.shape
		current_emb2d = current_emb.view(-1, feats_emb) # transfer 3d (batch * candidate * emb) to 2d ((batch*candidate)*emb) 
		history_emb2d = history_emb.unsqueeze(1).repeat(1,item_num,1,1).view(-1,max_len,his_emb)
		hislens2d = history_lengths.unsqueeze(1).repeat(1,item_num).view(-1)
		user_his_emb2d = self.attention(current_emb2d, history_emb2d, hislens2d,mask_mat,softmax_stag=False)
  
		din_output = torch.cat([user_his_emb2d, user_his_emb2d*current_emb2d, all_context.view(batch_size*item_num,-1) ],dim=-1)
		din_output = self.dnn_mlp_layers(din_output)
		return din_output.squeeze(dim=-1).view(batch_size, item_num)


class DINCTR(ContextSeqCTRModel, DINBase):
	reader = 'ContextSeqReader'
	runner = 'CTRRunner'
	extra_log_args = ['emb_size','att_layers','add_historical_situations']
	
	@staticmethod
	def parse_model_args(parser):
		parser = DINBase.parse_model_args_din(parser)
		return ContextSeqCTRModel.parse_model_args(parser)
	
	def __init__(self, args, corpus):
		ContextSeqCTRModel.__init__(self, args, corpus)
		self._define_init(args,corpus)

	def forward(self, feed_dict):
		out_dict = DINBase.forward(self,feed_dict)
		out_dict['prediction'] = out_dict['prediction'].view(-1).sigmoid()
		out_dict['label'] = feed_dict['label'].view(-1)
		return out_dict

class DINTopK(ContextSeqModel, DINBase):
	reader = 'ContextSeqReader'
	runner = 'BaseRunner'
	extra_log_args = ['emb_size','att_layers','add_historical_situations']
	
	@staticmethod
	def parse_model_args(parser):
		parser = DINBase.parse_model_args_din(parser)
		return ContextSeqModel.parse_model_args(parser)
	
	def __init__(self, args, corpus):
		ContextSeqModel.__init__(self, args, corpus)
		self._define_init(args,corpus)

	def forward(self, feed_dict):
		return DINBase.forward(self, feed_dict)
