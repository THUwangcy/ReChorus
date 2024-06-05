# -*- coding: UTF-8 -*-
# @Author : Zhiyu He
# @Email  : hezy22@mails.tsinghua.edu.cn

""" SDIM
Reference: 
	'Sampling is all you need on modeling long-term user behaviors for CTR prediction.', Cao, et al. , CIKM2022.
Implementation reference: FuxiCTR
	https://github.com/reczoo/FuxiCTR/blob/main/model_zoo/SDIM/src/SDIM.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as fn
import numpy as np
import pandas as pd

from models.BaseContextModel import ContextSeqModel, ContextSeqCTRModel
from models.context_seq.ETA import *
from utils.layers import MultiHeadTargetAttention, MLP_Block 

class SDIMBase(ETABase):
	@staticmethod
	def parse_model_args_SDIM(parser):
		'''
			Reuse ETA's args. The following args from ETA is not used for SDIM:
				retrieval_k. 
		'''
		return ETABase.parse_model_args_eta(parser)
	
	def _define_params_SDIM(self):
		# embedding
		self.embedding_dict = nn.ModuleDict()
		for f in self.user_context+self.item_context+self.situation_context:
			self.embedding_dict[f] = nn.Embedding(self.feature_max[f],self.vec_size) if f.endswith('_c') or f.endswith('_id') else\
					nn.Linear(1,self.vec_size,bias=False)
		self.powers_of_two = nn.Parameter(torch.tensor([2.0 ** i for i in range(self.hash_bits)]),requires_grad=False)
		pre_feature_num=0
		# short
		if self.recent_k > 0:
			self.short_attention = nn.ModuleList()
			for target_field in self.short_target_field:
				if type(target_field) == tuple:
					input_dim = self.vec_size * len(target_field)
					pre_feature_num+=len(target_field)
				else:
					input_dim = self.vec_size
					pre_feature_num+=1
				self.short_attention.append(MultiHeadTargetAttention(
					input_dim, self.attention_dim, self.num_heads,
					self.attention_dropout, self.use_scale, self.use_qkvo))
		# long
		if self.history_max > self.recent_k:
			self.random_rotations = nn.ParameterList()
			for target_field in self.long_target_field:
				if type(target_field) == tuple:
					input_dim = self.vec_size * len(target_field)
					pre_feature_num+=len(target_field)
				else:
					input_dim = self.vec_size
					pre_feature_num+=1
				self.random_rotations.append(nn.Parameter(torch.randn(input_dim,
									self.num_hashes, self.hash_bits), requires_grad=False))
		# Whether to use output activation
		# pre_feature_num = len(list(self.short_sequence_field)) + len(list(self.long_sequence_field))
		# dnn
		self.dnn = MLP_Block(input_dim = pre_feature_num * self.vec_size,
							 output_dim=1,
							 hidden_units=eval(self.dnn_hidden_units),
							 hidden_activations=self.dnn_activations,
							 dropout_rates=self.net_dropout,
							 batch_norm=self.batch_norm)

	def _define_init(self, args, corpus):
		self._define_hyper_params_eta(args, corpus)
		self._define_params_SDIM()
		self.apply(self.init_weights)
	
	def long_interest_attention(self, feature_emb_dict, mask, feature_emb):
		# long interest attention
		for idx, (target_field, sequence_field) in enumerate(zip(self.long_target_field, 
																 self.long_sequence_field)):
			target_emb = self.concat_embedding(target_field, feature_emb_dict) # .flatten(start_dim=-2)
			sequence_emb = self.concat_embedding(sequence_field, feature_emb_dict) # .flatten(start_dim=-2)
			target_emb_flatten = target_emb.view(-1,target_emb.size(-1))
			sequence_emb_flatten = sequence_emb.unsqueeze(1).repeat(1,target_emb.size(1),1,1).view(
	   					-1,sequence_emb.size(1),sequence_emb.size(2))
			long_interest_emb_flatten = self.lsh_attention(self.random_rotations[idx], 
													target_emb_flatten, sequence_emb_flatten)
			long_interest_emb = long_interest_emb_flatten.view(target_emb.shape) # batch * item num * embedding
			feature_emb.append(long_interest_emb)
		return feature_emb
	
	def forward(self, feed_dict):
		hislens = feed_dict['lengths']
		# mask = torch.arange(feed_dict['history_item_id'].shape[1], device=self.device)[None, :] < hislens[:, None] # batch size * history length
		indices = torch.arange(feed_dict['history_item_id'].shape[1]-1, -1, -1, device=feed_dict['history_item_id'].device)[None, :]
		mask_short = (indices < hislens[:, None]) & (indices <= self.recent_k)
		mask_long = (indices < hislens[:, None]) & (indices > self.recent_k)
		feature_emb_dict = self.get_embeddings_ETA(feed_dict)
		if self.recent_k>0:
			feature_emb = self.short_interest_attention(feature_emb_dict, mask_short)
		else:
			feature_emb = []
		if self.history_max > self.recent_k:
			feature_emb = self.long_interest_attention(feature_emb_dict, mask_long, feature_emb)
		feature_emb = torch.cat(feature_emb,dim=-1)
		# DNN
		batch_size, item_num, emb_dim = feature_emb.shape
		predictions = self.dnn(feature_emb.view(-1,emb_dim)).view(batch_size, item_num, -1).squeeze(-1)
		return {'prediction':predictions}


	def lsh_attention(self, random_rotations, target_item, history_sequence):
		""" References
		FuxiCTR - https://github.com/reczoo/FuxiCTR/blob/main/model_zoo/SDIM/src/SDIM.py
		"""
		if not self.reuse_hash:
			random_rotations = torch.randn(target_item.size(1), self.num_hashes, 
										   self.hash_bits, device=target_item.device)
		target_bucket = self.lsh_hash(history_sequence, random_rotations)
		sequence_bucket = self.lsh_hash(target_item.unsqueeze(1), random_rotations)
		bucket_match = (sequence_bucket - target_bucket).permute(2, 0, 1) # num_hashes x B x seq_len
		collide_mask = (bucket_match == 0).float()
		hash_index, collide_index = torch.nonzero(collide_mask.flatten(start_dim=1), as_tuple=True)
		offsets = collide_mask.sum(dim=-1).long().flatten().cumsum(dim=0)
		attn_out = fn.embedding_bag(collide_index, history_sequence.view(-1, target_item.size(1)), 
								   offsets, mode='sum') # (num_hashes x B) x d
		attn_out = attn_out.view(self.num_hashes, -1, target_item.size(1)).mean(dim=0) # B x d
		return attn_out

class SDIMCTR(ContextSeqCTRModel, SDIMBase):
	reader = 'ContextSeqReader'
	runner = 'CTRRunner'
	extra_log_args = ['emb_size','add_historical_situations'] # add_historical_situation: this parameter didn't use
	
	@staticmethod
	def parse_model_args(parser):
		parser = SDIMBase.parse_model_args_SDIM(parser)
		return ContextSeqCTRModel.parse_model_args(parser)
	
	def __init__(self, args, corpus):
		ContextSeqCTRModel.__init__(self, args, corpus)
		self._define_init(args,corpus)

	def forward(self, feed_dict):
		out_dict = SDIMBase.forward(self,feed_dict)
		out_dict['prediction'] = out_dict['prediction'].view(-1).sigmoid()
		out_dict['label'] = feed_dict['label'].view(-1)
		return out_dict

class SDIMTopK(ContextSeqModel, SDIMBase):
	reader = 'ContextSeqReader'
	runner = 'BaseRunner'
	extra_log_args = ['emb_size','add_historical_situations']
	
	@staticmethod
	def parse_model_args(parser):
		parser = SDIMBase.parse_model_args_SDIM(parser)
		return ContextSeqModel.parse_model_args(parser)
	
	def __init__(self, args, corpus):
		ContextSeqModel.__init__(self, args, corpus)
		self._define_init(args,corpus)

	def forward(self, feed_dict):
		return SDIMBase.forward(self, feed_dict)
