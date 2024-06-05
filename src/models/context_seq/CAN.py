# -*- coding: UTF-8 -*-
# @Author : Jiayu Li 
# @Email  : jy-li20@mails.tsinghua.edu.cn

'''
Reference:
 	CAN: feature co-action network for click-through rate prediction.
	Bian, Weijie, et al. 
  	Proceedings of the fifteenth ACM international conference on web search and data mining. 2022.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn.functional as fn
import pandas as pd

from models.BaseContextModel import ContextSeqModel, ContextSeqCTRModel
from models.context_seq.DIEN import *

class CANBase(DIENBase):
	@staticmethod
	def parse_model_args_can(parser):
		parser.add_argument('--induce_vec_size',type=int,default=512,
                      help='the size of the induce feature co-action vector')
		parser.add_argument('--orders',type=int,default=1,
                      help='numbers of orders of the feature co-action vector')
		parser.add_argument('--co_action_layers',type=str,default='[4,4]',
                      help='layers for the micro-MLP in co-action module')
		return DIENBase.parse_model_args_dien(parser)

	def _define_init(self, args, corpus):
		self._define_init_dien(args, corpus)
		self.induce_vec_size = args.induce_vec_size
		self.orders = args.orders
		self.co_action_layers = eval(args.co_action_layers)
		pre_size = self.embedding_size*self.orders
		co_action_nums = 0
		for layer_size in self.co_action_layers:
			co_action_nums += pre_size*layer_size + layer_size
			pre_size = layer_size
		assert self.induce_vec_size>=co_action_nums
		inp_shape = sum(self.co_action_layers) * ((len(self.situation_context)+1)+ 1)
		self.fcn_embedding_size = self.embedding_size*(len(self.user_context)+len(self.situation_context)+len(self.item_context))+\
						self.gru_emb_size*3 + inp_shape

		self._define_params_CAN()
		self.apply(self.init_weights)

	def _define_params_CAN(self):
		self._define_params_DIEN()
		self.item_embedding_induce = nn.Embedding(self.feature_max['item_id'], self.induce_vec_size)
		self.activation = nn.Tanh()
	
	def forward(self, feed_dict):
		item_ids = feed_dict['item_id'] # B * item num
		user_ids = feed_dict['user_id'] # B * item num
		history_item_ids = feed_dict['history_item_id']

		hislens = feed_dict['lengths'] # B
		mask = 	torch.arange(history_item_ids.shape[1])[None,:].to(self.device) < hislens[:,None]

		# embedding
		target_emb, history_emb, user_emb, context_emb = self.get_all_embeddings(feed_dict)
		item_ids_induce = self.item_embedding_induce(item_ids)
		user_ids_emb = self.embedding_dict['user_id'](user_ids)
		item_his_emb = self.embedding_dict['item_id'](history_item_ids)

		# co-action between user and item
		ui_coaction = self.gen_coaction(item_ids_induce, user_ids_emb.unsqueeze(dim=1),)
		# co-cation between situation context and item
		ci_coaction = []
		for s_feature in range(len(self.situation_context)):
			ci_coaction.append(self.gen_coaction(item_ids_induce, context_emb[:,s_feature*self.embedding_size:(s_feature+1)*self.embedding_size].unsqueeze(dim=1)))
		ci_coaction = torch.cat(ci_coaction,dim=-1)
		# history co-cation layer
		his_coaction = self.gen_his_coation(item_ids_induce, item_his_emb, mask)
 	
		# dien
		dien_inp, out_dict = self._get_inp(feed_dict)
		all_coaction = torch.cat([ui_coaction,ci_coaction,his_coaction,dien_inp,],dim=-1)
		logit = self.fcn_net(all_coaction).squeeze(dim=-1)
		out_dict['prediction'] = logit
		return out_dict 

	def gen_coaction(self, induction, feed):
		# induction: B * item num * induce vec size; feed: B * 1 * feed vec size
		B, item_num, _ = induction.shape
		feed_orders = []
		for i in range(self.orders):
			feed_orders.append(feed**(i+1))
		feed_orders = torch.cat(feed_orders,dim=-1) # B * 1 * (feed vec size * order)

		weight, bias = [], []
		pre_size = feed_orders.shape[-1]
		start_dim = 0
		for layer in self.co_action_layers:
			weight.append(induction[:,:,start_dim:start_dim+pre_size*layer].view(B,item_num,pre_size,layer))
			start_dim += pre_size*layer
			bias.append(induction[:,:,start_dim:start_dim+layer]) # B * item num * layer
			start_dim += layer
			pre_size = layer

		outputs = []
		hidden_state = feed_orders.repeat(1,item_num,1).unsqueeze(2)
		for layer_idx in range(len(self.co_action_layers)):
			hidden_state = self.activation(torch.matmul(hidden_state, weight[layer_idx]) + bias[layer_idx].unsqueeze(2))
			outputs.append(hidden_state.squeeze(2))
		outputs = torch.cat(outputs,dim=-1)
		return outputs
			
	def gen_his_coation(self, induction, feed, mask):
		# induction: B * item num * induce vec size; feed_his: B * his * feed vec size
		B, item_num, _ = induction.shape
		max_len = feed.shape[1]
		
		feed_orders = []
		for i in range(self.orders):
			feed_orders.append(feed**(i+1))
		feed_orders = torch.cat(feed_orders,dim=-1) # B * his * (feed vec size * order)

		weight, bias = [], []
		pre_size = feed_orders.shape[-1]
		start_dim = 0
		for layer in self.co_action_layers:
			weight.append(induction[:,:,start_dim:start_dim+pre_size*layer].view(B,item_num,pre_size,layer))
			start_dim += pre_size*layer
			bias.append(induction[:,:,start_dim:start_dim+layer]) # B * item num * layer
			start_dim += layer
			pre_size = layer
	
		outputs = []
		hidden_state = feed_orders.unsqueeze(2).repeat(1,1,item_num,1).unsqueeze(3)
		for layer_idx in range(len(self.co_action_layers)):
			# weight: B * item num * pre size * size, hidden: B * his len * item num * 1 * pre size
			hidden_state = self.activation(torch.matmul(hidden_state, 
								weight[layer_idx].unsqueeze(1)) + 
				   				bias[layer_idx].unsqueeze(1).unsqueeze(3)) # B * his len * item num * 1 * size
			outputs.append((hidden_state.squeeze(3)*mask[:,:,None,None]).sum(dim=1)/mask.sum(dim=-1)[:,None,None]) # B * item num * size
		outputs = torch.cat(outputs,dim=-1)
		return outputs
 	
class CANTopK(ContextSeqModel, CANBase):
	reader, runner = 'ContextSeqReader', 'BaseRunner'
	extra_log_args = ['emb_size','evolving_gru_type','fcn_hidden_layers']
	
	@staticmethod
	def parse_model_args(parser):
		parser = CANBase.parse_model_args_can(parser)
		return ContextSeqModel.parse_model_args(parser)

	def __init__(self, args, corpus):
		ContextSeqModel.__init__(self, args, corpus)
		self._define_init(args,corpus)

	def forward(self, feed_dict):
		return CANBase.forward(self, feed_dict)

	def loss(self, out_dict):
		loss = ContextSeqModel.loss(self, out_dict)
		if self.alpha_aux>0:
			loss += self.alpha_aux*self.aux_loss(out_dict)
		return loss

	class Dataset(DIENTopK.Dataset):
		def _get_feed_dict(self, index):
			feed_dict = super()._get_feed_dict(index)
			return feed_dict
	
class CANCTR(ContextSeqCTRModel, CANBase):
	reader = 'ContextSeqReader'
	runner = 'CTRRunner'
	extra_log_args = ['emb_size','evolving_gru_type']

	@staticmethod
	def parse_model_args(parser):
		parser = CANBase.parse_model_args_can(parser)
		return ContextSeqCTRModel.parse_model_args(parser)

	def __init__(self, args, corpus):
		ContextSeqCTRModel.__init__(self, args, corpus)
		self._define_init(args,corpus)

	def forward(self, feed_dict):
		out_dict = CANBase.forward(self,feed_dict)
		out_dict['prediction'] = out_dict['prediction'].view(-1).sigmoid()
		out_dict['label'] = feed_dict['label'].view(-1)
		return out_dict

	def loss(self, out_dict):
		loss = ContextSeqCTRModel.loss(self, out_dict)
		if self.alpha_aux>0:
			loss += self.alpha_aux*self.aux_loss(out_dict)
		return loss

	class Dataset(DIENCTR.Dataset):
		def _get_feed_dict(self, index):
			feed_dict = super()._get_feed_dict(index)
			return feed_dict