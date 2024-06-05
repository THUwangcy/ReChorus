# -*- coding: UTF-8 -*-
# @Author : Zhiyu He
# @Email  : hezy22@mails.tsinghua.edu.cn

""" ETA
Reference: 
	Chen Q, Pei C, Lv S, et al. End-to-end user behavior retrieval in click-through rateprediction model[J]. 
		arXiv preprint arXiv:2108.04468, 2021.
Implementation reference: FuxiCTR
	https://github.com/reczoo/FuxiCTR/blob/main/model_zoo/ETA/src/ETA.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as fn
import numpy as np
import pandas as pd

from models.BaseContextModel import ContextSeqModel, ContextSeqCTRModel
from utils.layers import MultiHeadTargetAttention, MLP_Block  

class ETABase(object):
	@staticmethod
	def parse_model_args_eta(parser):
		parser.add_argument('--emb_size', type=int, default=64,
                      help="Size of embedding vectors.")
		# dnn
		parser.add_argument('--dnn_hidden_units',type=str,default='[128,64]',
                      help="Size of each hidden layer.") # [512,128,64]
		parser.add_argument('--dnn_activations',type=str,default='ReLU',
                      help="The activation function to be used in DNN. ")
		parser.add_argument('--net_dropout',type=float,default=0, help="Dropout rate for DNN.")
		parser.add_argument('--batch_norm',type=int,default=0, help="Whether to use batch_norm or not.")
		# attention
		parser.add_argument('--attention_dim',type=int,default=64,
                      help="Size of attention hidden space.")
		parser.add_argument('--num_heads',type=int,default=1, 
                      help="Number of attention heads.")
		parser.add_argument('--use_scale',type=int,default=1, 
                      help="Wheter to use scaling factor when calculating attention weights.")
		parser.add_argument('--attention_dropout',type=float,default=0, 
                      help="Dropout rate for attention.")
		parser.add_argument('--use_qkvo',type=int,default=True, 
                      		help="Whether to apply separate linear transformations for multi-head target attention.")
		parser.add_argument('--retrieval_k', type=int, default=5,
                      help="Retrieve top-k similar items from long-term user behavior sequence.")
		# hash
		parser.add_argument('--reuse_hash',type=int,default=1, 
                      help="Wheter to use hash for long interest attention.")
		parser.add_argument('--num_hashes',type=int,default=1,
                      help="Number of separate hashes.")
		parser.add_argument('--hash_bits',type=int,default=4,
                      help="Number of bits used for each hash.")
		# long & short fields
		parser.add_argument('--short_target_field',type=str,default='["item_id"]', # '["item_id","i_category_c","c_day_f"]'
					  help='Select features.')
		parser.add_argument('--short_sequence_field',type=str,default='["history_item_id"]', # '["history_item_id","history_i_category_c","history_c_day_f"]',
                      help='Select features, short_sequence_field should match with short_target_field') 
		parser.add_argument('--long_target_field',type=str,default='["item_id"]', # '["item_id","i_category_c","c_day_f"]',
                      help='Select features.')
		parser.add_argument('--long_sequence_field',type=str,default='["history_item_id"]', #'["history_item_id","history_i_category_c","history_c_day_f"]',
                      help='Select features, long_sequence_field should match with long_target_field')
		parser.add_argument('--recent_k',type=int,default=5, 
                      help='Define the threshold for short term and long term hisotry behavior, should be less than the history_max.')
		return parser

	def _define_hyper_params_eta(self, args, corpus):
		self.user_context = ['user_id']+corpus.user_feature_names
		self.item_context = ['item_id']+corpus.item_feature_names
		self.situation_context = corpus.situation_feature_names
		self.item_feature_num = len(corpus.item_feature_names)+1
		self.user_feature_num = len(corpus.user_feature_names)+1
		self.vec_size = args.emb_size

		# attention
		self.attention_dim = args.attention_dim
		self.num_heads = args.num_heads
		self.attention_dropout = args.attention_dropout
		self.use_scale = args.use_scale
		self.use_qkvo = args.use_qkvo
  
		self.retrieval_k = args.retrieval_k
		# Hash
		self.reuse_hash = args.reuse_hash # True/False
		self.num_hashes = args.num_hashes
		self.hash_bits = args.hash_bits
		# target
		self.short_target_field = eval(args.short_target_field)
		if type(self.short_target_field) != list:
			self.short_target_field = [self.short_target_field]
		self.short_sequence_field = eval(args.short_sequence_field)
		if type(self.short_sequence_field) != list:
			self.short_sequence_field = [self.short_sequence_field]
		self.long_target_field = eval(args.long_target_field)
		if type(self.long_target_field) != list:
			self.long_target_field = [self.long_target_field]
		self.long_sequence_field = eval(args.long_sequence_field)
		if type(self.long_sequence_field) != list:
			self.long_sequence_field = [self.long_sequence_field]
		assert len(self.short_target_field) == len(self.short_sequence_field) \
			   and len(self.long_target_field) == len(self.long_sequence_field), \
			   "Config error: target_field mismatches with sequence_field."
      
		self.recent_k = args.recent_k
		assert self.recent_k <= self.history_max

		# dnn
		self.dnn_hidden_units = args.dnn_hidden_units
		self.dnn_activations = args.dnn_activations
		self.net_dropout = args.net_dropout
		self.batch_norm = args.batch_norm

	def _define_init(self, args,corpus):
		self._define_hyper_params_eta(args, corpus)
		self._define_params_ETA()
		self.apply(self.init_weights)

	def _define_params_ETA(self):
		# embedding
		self.embedding_dict = nn.ModuleDict()
		for f in self.user_context+self.item_context+self.situation_context:
			self.embedding_dict[f] = nn.Embedding(self.feature_max[f],self.vec_size) if f.endswith('_c') or f.endswith('_id') else\
					nn.Linear(1,self.vec_size,bias=False)
		self.powers_of_two = nn.Parameter(torch.tensor([2.0 ** i for i in range(self.hash_bits)]),requires_grad=False)
		# short
		self.short_attention = nn.ModuleList()
		pre_feature_num=0
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
			self.long_attention = nn.ModuleList()
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
				self.long_attention.append(MultiHeadTargetAttention(
					input_dim, self.attention_dim, self.num_heads,
					self.attention_dropout, self.use_scale,self.use_qkvo))

		# Whether to use output activation
		# pre_feature_num = len(list(self.short_sequence_field)) + len(list(self.long_sequence_field))
		# dnn
		self.dnn = MLP_Block(input_dim = pre_feature_num * self.vec_size,
							 output_dim=1,
							 hidden_units=eval(self.dnn_hidden_units),
							 hidden_activations=self.dnn_activations,
							 dropout_rates=self.net_dropout,
							 batch_norm=self.batch_norm)

	def short_interest_attention(self, feature_emb_dict, mask):
		# short interest attention
		feature_emb = []
		for idx, (target_field, sequence_field) in enumerate(zip(self.short_target_field, 
																 self.short_sequence_field)):
			target_emb = self.concat_embedding(target_field, feature_emb_dict) # batch * item num * embedding
			sequence_emb = self.concat_embedding(sequence_field, feature_emb_dict) # batch * his_len/item_num * embedding
			target_emb_flatten = target_emb.view(-1,target_emb.size(-1))
			sequence_emb_flatten = sequence_emb.unsqueeze(1).repeat(1,target_emb.size(1),1,1).view(  # batch * item num * his_len * embedding
	   					-1,sequence_emb.size(1),sequence_emb.size(2))
			mask_flatten = mask.unsqueeze(1).repeat(1,target_emb.size(1),1).view(-1,sequence_emb_flatten.size(1))
			short_interest_emb_flatten = self.short_attention[idx](target_emb_flatten, sequence_emb_flatten, mask_flatten)
			short_interest_emb = short_interest_emb_flatten.view(target_emb.shape) # batch * item num * embedding
			feature_emb.append(short_interest_emb)
		return feature_emb

	def long_interest_attention(self, feature_emb_dict, mask, feature_emb):
		# long interest attention
		for idx, (target_field, sequence_field) in enumerate(zip(self.long_target_field, 
																 self.long_sequence_field)):
			target_emb = self.concat_embedding(target_field, feature_emb_dict) # .flatten(start_dim=-2)
			sequence_emb = self.concat_embedding(sequence_field, feature_emb_dict) # .flatten(start_dim=-2)
			target_emb_flatten = target_emb.view(-1,target_emb.size(-1))
			sequence_emb_flatten = sequence_emb.unsqueeze(1).repeat(1,target_emb.size(1),1,1).view(
	   					-1,sequence_emb.size(1),sequence_emb.size(2))
			mask_flatten = mask.unsqueeze(1).repeat(1,target_emb.size(1),1).view(-1,sequence_emb_flatten.size(1))
			topk_emb, topk_mask = self.topk_retrieval(self.random_rotations[idx], 
                                             target_emb_flatten, sequence_emb_flatten, mask_flatten, self.retrieval_k)
			long_interest_emb_flatten = self.long_attention[idx](target_emb_flatten, topk_emb, topk_mask)
			long_interest_emb = long_interest_emb_flatten.view(target_emb.shape)
			
			feature_emb.append(long_interest_emb)
		return feature_emb
	
	def forward(self, feed_dict):
		hislens = feed_dict['lengths']
		# mask = torch.arange(feed_dict['history_item_id'].shape[1], device=self.device)[None, :] < hislens[:, None] # batch size * history length
		indices = torch.arange(feed_dict['history_item_id'].shape[1]-1, -1, -1, device=feed_dict['history_item_id'].device)[None, :]
		mask_short = (indices < hislens[:, None]) & (indices <= self.recent_k)
		mask_long = (indices < hislens[:, None]) & (indices > self.recent_k)

		feature_emb_dict = self.get_embeddings_ETA(feed_dict)

		feature_emb = self.short_interest_attention(feature_emb_dict, mask_short)
		if self.history_max > self.recent_k:
			feature_emb = self.long_interest_attention(feature_emb_dict, mask_long, feature_emb)
		feature_emb = torch.cat(feature_emb,dim=-1)
		# DNN
		batch_size, item_num, emb_dim = feature_emb.shape
		predictions = self.dnn(feature_emb.view(-1,emb_dim)).view(batch_size, item_num, -1).squeeze(-1)
		return {'prediction':predictions}

	def get_embeddings_ETA(self, feed_dict):
		_, item_num = feed_dict['item_id'].shape
		_, his_lens = feed_dict['history_item_id'].shape
		feature_emb_dict = dict()
		for f_all in self.feature_max:
			if f_all=='user_id' or f_all.startswith('u_'):
				f_list = [f_all, 'new_history_'+f_all]
			elif f_all=='item_id' or f_all.startswith('i_'):
				f_list = [f_all, 'history_'+f_all]
			else:
				if self.add_historical_situations: 
					f_list = [f_all, 'history_'+f_all]
				else:
					f_list = [f_all, 'new_history_'+f_all]
			for f in f_list:
				if f.startswith('new_'):
					tmp_f = f.split('new_history_')[1]
					feature_emb_dict[f.split('new_')[1]] = self.embedding_dict[f_all](feed_dict[tmp_f]) if tmp_f.endswith('_c') or tmp_f.endswith('_id') \
										else self.embedding_dict[f_all](feed_dict[tmp_f].float().unsqueeze(-1))
					feature_emb_dict[f.split('new_')[1]] = feature_emb_dict[f.split('new_')[1]] if len(feature_emb_dict[f.split('new_')[1]].shape)==3 else feature_emb_dict[f.split('new_')[1]].unsqueeze(dim=-2).repeat(1, his_lens, 1)

				else:
					feature_emb_dict[f] = self.embedding_dict[f_all](feed_dict[f]) if f.endswith('_c') or f.endswith('_id') \
										else self.embedding_dict[f_all](feed_dict[f].float().unsqueeze(-1))
					feature_emb_dict[f] = feature_emb_dict[f] if len(feature_emb_dict[f].shape)==3 else feature_emb_dict[f].unsqueeze(dim=-2).repeat(1, item_num, 1)
		return feature_emb_dict

	def concat_embedding(self, field, feature_emb_dict):
		if type(field) == tuple:
			emb_list = [feature_emb_dict[f] for f in field]
			return torch.cat(emb_list, dim=-1)
		else:
			return feature_emb_dict[field]

	def topk_retrieval(self, random_rotations, target_item, history_sequence, mask, topk=5):
		""" Reference:
			FxiCTR - https://github.com/reczoo/FuxiCTR/blob/main/model_zoo/ETA/src/ETA.py
		"""
		if not self.reuse_hash:
			random_rotations = torch.randn(target_item.size(1), self.hash_bits, device=target_item.device)
		target_hash = self.lsh_hash(history_sequence, random_rotations) # B * num_item, his_lens), hash_bits
		sequence_hash = self.lsh_hash(target_item.unsqueeze(1), random_rotations)
		hash_sim = -torch.abs(sequence_hash - target_hash).sum(dim=-1)
		hash_sim = hash_sim.masked_fill_(mask.float() == 0, -self.hash_bits)
		topk = min(topk, hash_sim.shape[1])
		topk_index = hash_sim.topk(topk, dim=1, largest=True, sorted=True)[1]
		topk_emb = torch.gather(history_sequence, 1, 
                                topk_index.unsqueeze(-1).expand(-1, -1, history_sequence.shape[-1]))
		topk_mask = torch.gather(mask, 1, topk_index)
		return topk_emb, topk_mask

	def lsh_hash(self, vecs, random_rotations):
		""" Reference:
			https://github.com/brc7/tensorflow-lsh-functions/blob/main/lsh_functions.py
			
			Input: vecs (shape B x seq_len x d)
			Output: hash_bucket (shape B x seq_len x num_hashes)
		"""
		rotated_vecs = torch.einsum("bld,dht->blht", vecs, random_rotations) # B x seq_len x num_hashes x hash_bits
		hash_code = torch.relu(torch.sign(rotated_vecs))
		hash_bucket = torch.matmul(hash_code, self.powers_of_two.unsqueeze(-1)).squeeze(-1)
		return hash_bucket

class ETACTR(ContextSeqCTRModel, ETABase):
	reader = 'ContextSeqReader'
	runner = 'CTRRunner'
	extra_log_args = ['emb_size','add_historical_situations'] # add_historical_situation: this parameter didn't use
	
	@staticmethod
	def parse_model_args(parser):
		parser = ETABase.parse_model_args_eta(parser)
		return ContextSeqCTRModel.parse_model_args(parser)
	
	def __init__(self, args, corpus):
		ContextSeqCTRModel.__init__(self, args, corpus)
		self._define_init(args,corpus)

	def forward(self, feed_dict):
		out_dict = ETABase.forward(self,feed_dict)
		out_dict['prediction'] = out_dict['prediction'].view(-1).sigmoid()
		out_dict['label'] = feed_dict['label'].view(-1)
		return out_dict

class ETATopK(ContextSeqModel, ETABase):
	reader = 'ContextSeqReader'
	runner = 'BaseRunner'
	extra_log_args = ['emb_size','add_historical_situations']
	
	@staticmethod
	def parse_model_args(parser):
		parser = ETABase.parse_model_args_eta(parser)
		return ContextSeqModel.parse_model_args(parser)
	
	def __init__(self, args, corpus):
		ContextSeqModel.__init__(self, args, corpus)
		self._define_init(args,corpus)

	def forward(self, feed_dict):
		return ETABase.forward(self, feed_dict)