# -*- coding: UTF-8 -*-
# @Author : Jiayu Li 
# @Email  : jy-li20@mails.tsinghua.edu.cn

""" DeepFM
Reference:
	'DeepFM: A Factorization-Machine based Neural Network for CTR Prediction', Guo et al., IJCAI 2017.
"""
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from utils import layers
from models.context.WideDeep import WideDeepCTR, WideDeepTopK
from models.context.WideDeep import WideDeepBase

class DeepFMBase(WideDeepBase):
	def forward(self, feed_dict):
		context_vectors, linear_vectors = self._get_embeddings_FM(feed_dict)
		# FM
		fm_vectors = 0.5 * (context_vectors.sum(dim=-2).pow(2) - context_vectors.pow(2).sum(dim=-2))
		fm_prediction = fm_vectors.sum(dim=-1) + linear_vectors
		# Deep
		deep_prediction = self.deep_layers(context_vectors.flatten(start_dim=-2)).squeeze(dim=-1)
		
		predictions = fm_prediction + deep_prediction
		return {'prediction':predictions}

class DeepFMCTR(WideDeepCTR, DeepFMBase):
	reader, runner = 'ContextReader', 'CTRRunner'
	extra_log_args = ['emb_size','layers','loss_n']
    
	def __init__(self, args, corpus):
		WideDeepCTR.__init__(self, args, corpus)

	def forward(self, feed_dict):
		out_dict = DeepFMBase.forward(self, feed_dict)
		out_dict['prediction'] = out_dict['prediction'].view(-1).sigmoid()
		out_dict['label'] = feed_dict['label'].view(-1)
		return out_dict

class DeepFMTopK(WideDeepTopK, DeepFMBase):
	reader, runner = 'ContextReader','BaseRunner'
	extra_log_args = ['emb_size','layers','loss_n']

	def __init__(self, args, corpus):
		WideDeepTopK.__init__(self, args, corpus)

	def forward(self, feed_dict):
		return DeepFMBase.forward(self, feed_dict)
