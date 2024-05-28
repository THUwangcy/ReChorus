# -*- coding: UTF-8 -*-
# @Author : Jiayu Li 
# @Email  : jy-li20@mails.tsinghua.edu.cn

"""
References:
	'Looking at CTR Prediction Again: Is Attention All You Need?', Cheng et al., SIGIR2021.
Implementation reference: FuxiCTR
	https://github.com/reczoo/FuxiCTR/blob/v2.0.1/model_zoo/SAM/src/SAM.py
"""
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import logging

from utils import layers
from models.BaseContextModel import ContextCTRModel, ContextModel

class SAMBase(object):
	@staticmethod
	def parse_model_args_sam(parser):
		parser.add_argument('--emb_size', type=int, default=64,
							help='Size of embedding vectors.')
		parser.add_argument('--interaction_type',type=str,default='SAM2E',
                      help='Way to interact different features, including SAM2A, SAM2E, SAM3A, SAM3E, SAM1.')
		parser.add_argument('--aggregation',type=str,default='concat',
                      	help='Way to aggregate different features, including concat, weighted_pooling, mean_pooling, sum_pooling')
		parser.add_argument('--num_layers',type=int,default=1,
                      help='Number of layers in SAM block.')
		parser.add_argument('--use_residual',type=int,default=0,
                      help='Whether to use residual connection in SAM block.')
		return parser

	def _define_init(self, args, corpus):
		self.embedding_dim = args.emb_size
		self.interaction_type = args.interaction_type
		self.aggregation = args.aggregation
		self.num_layers = args.num_layers
		self.use_residual = args.use_residual
		if self.interaction_type in ['SAM2A', 'SAM2E'] and not self.aggregation == 'concat':
			logging.warning('Aggregation is set to concat for SAM2!')
			self.aggregation = 'concat'
		if self.interaction_type == 'SAM1' and not self.aggregation == 'weighted_pooling':
			logging.warning('Aggreation is set to weighted_pooling for SAM1!')
			self.aggregation = 'weighted_pooling'
		self._define_params_sam()
		self.apply(self.init_weights)

	def _define_params_sam(self):
		self.block = SAMBlock(num_layers=self.num_layers, num_fields=len(self.context_features),
                        	  embedding_dim=self.embedding_dim, use_residual=self.use_residual,
                           	  interaction_type=self.interaction_type,aggregation=self.aggregation,
                              dropout=self.dropout)
		self.embedding_dict = nn.ModuleDict()
		for f in self.context_features:
			self.embedding_dict[f] = nn.Embedding(self.feature_max[f],self.embedding_dim) if f.endswith('_c') or f.endswith('_id') else\
					nn.Linear(1,self.embedding_dim,bias=False)
		if self.aggregation == 'concat' and self.interaction_type!='SAM1':
			if self.interaction_type in ['SAM2A','SAM2E']:
				self.output_layer = nn.Linear(self.embedding_dim*(len(self.context_features)**2), 1)
			else:
				self.output_layer = nn.Linear(self.embedding_dim*(len(self.context_features)),1)
		else:
			self.output_layer = nn.Linear(self.embedding_dim, 1)

	def forward(self, feed_dict):
		item_num = feed_dict['item_id'].shape[1]
		feature_embeddings = [self.embedding_dict[f](feed_dict[f]) if f.endswith('_c') or f.endswith('_id') 
						  else self.embedding_dict[f](feed_dict[f].float().unsqueeze(-1))
						  for f in self.context_features]
		feature_embeddings = torch.stack([v if len(v.shape)==3 else v.unsqueeze(dim=-2).repeat(1, item_num, 1) 
							for v in feature_embeddings], dim=-2) # batch size * item num * feature num * feature dim: 84,100,2,64
		interacted_features = self.block(feature_embeddings)
		predictions = self.output_layer(interacted_features)
		return {'prediction':predictions.squeeze(dim=-1)}

class SAMCTR(ContextCTRModel, SAMBase):
	reader, runner = 'ContextReader', 'CTRRunner'
	extra_log_args = ['emb_size','loss_n', 'interaction_type','aggregation']

	@staticmethod
	def parse_model_args(parser):
		parser = SAMBase.parse_model_args_sam(parser)
		return ContextCTRModel.parse_model_args(parser)

	def __init__(self, args, corpus):
		ContextCTRModel.__init__(self, args, corpus)
		self._define_init(args, corpus)

	def forward(self, feed_dict):
		out_dict = SAMBase.forward(self, feed_dict)
		out_dict['prediction'] = out_dict['prediction'].view(-1).sigmoid()
		out_dict['label'] = feed_dict['label'].view(-1)
		return out_dict
		
class SAMTopK(ContextModel, SAMBase):
	reader, runner = 'ContextReader', 'BaseRunner'
	extra_log_args = ['emb_size','loss_n', 'interaction_type','aggregation']
    
	@staticmethod
	def parse_model_args(parser):
		parser = SAMBase.parse_model_args_sam(parser)
		return ContextModel.parse_model_args(parser)

	def __init__(self, args, corpus):
		ContextModel.__init__(self, args, corpus)
		self._define_init(args,corpus)

	def forward(self, feed_dict):
		return SAMBase.forward(self, feed_dict)


'''
The following codes refer to: FuxiCTR
	https://github.com/reczoo/FuxiCTR/blob/v2.0.1/model_zoo/SAM/src/SAM.py
'''
class SAMBlock(nn.Module):
    def __init__(self, num_layers, num_fields, embedding_dim, use_residual=False, 
                 interaction_type="SAM2E", aggregation="concat", dropout=0):
        super().__init__()
        self.aggregation = aggregation
        self.interaction_type = interaction_type
        if self.aggregation == "weighted_pooling":
            self.weight = nn.Parameter(torch.ones(num_fields, 1))
        if self.interaction_type == "SAM1":
            self.layers = nn.ModuleList([nn.Identity()])
        elif self.interaction_type == "SAM2A":
            self.layers = nn.ModuleList([SAM2A(num_fields, embedding_dim, dropout)])
        elif self.interaction_type == "SAM2E":
            self.layers = nn.ModuleList([SAM2E(embedding_dim, dropout)])
        elif self.interaction_type == "SAM3A":
            self.layers = nn.ModuleList([SAM3A(num_fields, embedding_dim, use_residual, dropout) \
                                         for _ in range(num_layers)])
        elif self.interaction_type == "SAM3E":
            self.layers = nn.ModuleList([SAM3E(embedding_dim, use_residual, dropout) \
                                         for _ in range(num_layers)])
        else:
            raise ValueError("interaction_type={} not supported.".format(interaction_type))

    def forward(self, F):
        # F: batch size * item_num * feature num * emb dim
        batch_size, item_num, num_fields, emb_dim = F.shape
        F = F.view(-1, num_fields, emb_dim)
        for layer in self.layers:
            F = layer(F)
        if self.aggregation == "weighted_pooling":
            out = (F * self.weight).sum(dim=1)
        elif self.aggregation == "concat":
            out = F.flatten(start_dim=1)
        elif self.aggregation == "mean_pooling":
            out = F.mean(dim=1)
        elif self.aggregation == "sum_pooling":
            out = F.sum(dim=1)
        return out.view(batch_size,item_num,-1)

class SAM2A(nn.Module):
    def __init__(self, num_fields, embedding_dim, dropout=0):
        super(SAM2A, self).__init__()
        self.W = nn.Parameter(torch.ones(num_fields, num_fields, embedding_dim))
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else None

    def forward(self, F):
        S = torch.bmm(F, F.transpose(1, 2)) # b x f x f
        out = S.unsqueeze(-1) * self.W # b x f x f x d
        if self.dropout:
            out = self.dropout(out)
        return out

class SAM2E(nn.Module):
    def __init__(self, embedding_dim, dropout=0):
        super(SAM2E, self).__init__()
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else None

    def forward(self, F):
        S = torch.bmm(F, F.transpose(1, 2)) # b x f x f
        U = torch.einsum("bnd,bmd->bnmd", F, F) # b x f x f x d
        out = S.unsqueeze(-1) * U # b x f x f x d
        if self.dropout:
            out = self.dropout(out)
        return out

class SAM3A(nn.Module):
    def __init__(self, num_fields, embedding_dim, use_residual=True, dropout=0):
        super(SAM3A, self).__init__()
        self.W = nn.Parameter(torch.ones(num_fields, num_fields, embedding_dim)) # f x f x d
        self.K = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.use_residual = use_residual
        if use_residual:
            self.Q = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else None

    def forward(self, F):
        S = torch.bmm(F, self.K(F).transpose(1, 2)) # b x f x f
        out = (S.unsqueeze(-1) * self.W).sum(dim=2) # b x f x d
        if self.use_residual:
            out += self.Q(F)
        if self.dropout:
            out = self.dropout(out)
        return out

class SAM3E(nn.Module):
    def __init__(self, embedding_dim, use_residual=True, dropout=0):
        super(SAM3E, self).__init__()
        self.K = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.use_residual = use_residual
        if use_residual:
            self.Q = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else None

    def forward(self, F):
        S = torch.bmm(F, self.K(F).transpose(1, 2)) # b x f x f
        U = torch.einsum("bnd,bmd->bnmd", F, F) # b x f x f x d
        out = (S.unsqueeze(-1) * U).sum(dim=2) # b x f x d
        if self.use_residual:
            out += self.Q(F)
        if self.dropout:
            out = self.dropout(out)
        return out
