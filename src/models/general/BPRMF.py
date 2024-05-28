# -*- coding: UTF-8 -*-
# @Author  : Chenyang Wang
# @Email   : THUwangcy@gmail.com

""" BPRMF
Reference:
	"Bayesian personalized ranking from implicit feedback"
	Rendle et al., UAI'2009.
CMD example:
	python main.py --model_name BPRMF --emb_size 64 --lr 1e-3 --l2 1e-6 --dataset 'Grocery_and_Gourmet_Food'
"""

import torch.nn as nn

from models.BaseModel import GeneralModel
from models.BaseImpressionModel import ImpressionModel

class BPRMFBase(object):
	@staticmethod
	def parse_model_args(parser):
		parser.add_argument('--emb_size', type=int, default=64,
							help='Size of embedding vectors.')
		return parser

	def _base_init(self, args, corpus):
		self.emb_size = args.emb_size
		self._base_define_params()
		self.apply(self.init_weights)
	
	def _base_define_params(self):	
		self.u_embeddings = nn.Embedding(self.user_num, self.emb_size)
		self.i_embeddings = nn.Embedding(self.item_num, self.emb_size)

	def forward(self, feed_dict):
		self.check_list = []
		u_ids = feed_dict['user_id']  # [batch_size]
		i_ids = feed_dict['item_id']  # [batch_size, -1]

		cf_u_vectors = self.u_embeddings(u_ids)
		cf_i_vectors = self.i_embeddings(i_ids)

		prediction = (cf_u_vectors[:, None, :] * cf_i_vectors).sum(dim=-1)  # [batch_size, -1]
		u_v = cf_u_vectors.repeat(1,i_ids.shape[1]).view(i_ids.shape[0],i_ids.shape[1],-1)
		i_v = cf_i_vectors
		return {'prediction': prediction.view(feed_dict['batch_size'], -1), 'u_v': u_v, 'i_v':i_v}

class BPRMF(GeneralModel, BPRMFBase):
	reader = 'BaseReader'
	runner = 'BaseRunner'
	extra_log_args = ['emb_size', 'batch_size']

	@staticmethod
	def parse_model_args(parser):
		parser = BPRMFBase.parse_model_args(parser)
		return GeneralModel.parse_model_args(parser)

	def __init__(self, args, corpus):
		GeneralModel.__init__(self, args, corpus)
		self._base_init(args, corpus)

	def forward(self, feed_dict):
		out_dict =  BPRMFBase.forward(self, feed_dict)
		return {'prediction': out_dict['prediction']}

class BPRMFImpression(ImpressionModel, BPRMFBase):
	reader = 'ImpressionReader'
	runner = 'ImpressionRunner'
	extra_log_args = ['emb_size', 'batch_size']

	@staticmethod
	def parse_model_args(parser):
		parser = BPRMFBase.parse_model_args(parser)
		return ImpressionModel.parse_model_args(parser)

	def __init__(self, args, corpus):
		ImpressionModel.__init__(self, args, corpus)
		self._base_init(args, corpus)

	def forward(self, feed_dict):
		return BPRMFBase.forward(self, feed_dict)