import torch
import numpy as np
import torch.nn.functional as F
from typing import List
import yaml
import copy

from models.BaseModel import *
from models.BaseImpressionModel import *
from models.general import *
from models.sequential import *
from models.developing import *		


class RerankModel(ImpressionModel):
	reader='ImpressionReader'
	runner='ImpressionRunner'
	extra_log_args = ['tuneranker']

	@staticmethod
	def parse_model_args(parser):
		parser.add_argument('--ranker_name', type=str, default='BPRMF_test',
							help='Base ranker')
		parser.add_argument('--ranker_config_file', type=str, default='BPRMF_test.yaml',
							help='Base ranker config file')
		parser.add_argument('--ranker_model_file', type=str, default='BPRMF_test__MINDCTR__2__lr=0.001__l2=0.0__emb_size=64__batch_size=256.pt',
							help='Base ranker model file')
		parser.add_argument('--tuneranker', type=int, default=0,
							help='if 1, continue to train ranker')
		return ImpressionModel.parse_model_args(parser)

	def __init__(self, args, corpus):
		super().__init__(args, corpus)
		self.ranker_name = args.ranker_name
		self.ranker_config = args.ranker_config_file
		self.ranker_model = args.ranker_model_file
		self.tuneranker = args.tuneranker
		self.load_ranker(args, corpus)

	def load_ranker(self, args, corpus):
		corpus = corpus
		# config_path = './{}'.format(self.ranker_config)
		# model_path = './{}'.format(self.ranker_model)
		config_path = './model/{}Impression/{}'.format(self.ranker_name,self.ranker_config)
		model_path = './model/{}Impression/{}'.format(self.ranker_name,self.ranker_model)
		#read ranker config
		ranker_config_dict = dict()
		with open(config_path, "r", encoding="utf-8") as f:
			ranker_config_dict.update(
				yaml.load(f.read(), Loader=yaml.FullLoader)
			)
		#load ranker from model and get self.ranker_emb_size
		ranker_args = copy.deepcopy(args)
		for k, v in ranker_config_dict.items():
			if k != 'history_max':
				setattr(ranker_args, k, v)
		model_name = eval('{0}.{0}Impression'.format(self.ranker_name))
		self.ranker = model_name(ranker_args, corpus)
		self.ranker.device = ranker_args.device
		self.ranker.apply(self.ranker.init_weights)
		self.ranker.to(self.device)
		self.ranker_emb_size = ranker_args.emb_size
		self.ranker.load_model(model_path)
		if not self.tuneranker:
			for param in self.ranker.parameters():
				param.requires_grad = False

	class Dataset(ImpressionModel.Dataset):		
		# Collate a batch according to the list of feed dicts
		def collate_batch(self, feed_dicts: List[dict]) -> dict: # feed_dicts are a batch of dicts
			feed_dict = super().collate_batch(feed_dicts)
			feed_dict['batch_size'] = len(feed_dicts)
			predict_dict = self.model.ranker(utils.batch_to_gpu(feed_dict, self.model.device)) # pos+pad+neg+pad
			feed_dict['scores'] = predict_dict['prediction'] # [batch(or num_sequence),n_candidate]
			pos_mask = torch.arange(0, self.model.train_max_pos_item, device = self.model.device).type_as(feed_dict['pos_num']).unsqueeze(0).expand(feed_dict['batch_size'], self.model.train_max_pos_item).lt(feed_dict['pos_num'].unsqueeze(1))
			neg_mask = torch.arange(0, self.model.train_max_neg_item, device = self.model.device).type_as(feed_dict['neg_num']).unsqueeze(0).expand(feed_dict['batch_size'], self.model.train_max_neg_item).lt(feed_dict['neg_num'].unsqueeze(1))
			all_mask = torch.cat([pos_mask, neg_mask],dim = 1)
			feed_dict['padding_mask'] = ~all_mask
			feed_dict['scores'] = torch.where(all_mask == 1, feed_dict['scores'],-np.inf * torch.ones_like(feed_dict['scores']))
			_,temp = feed_dict['scores'].sort(dim = 1, descending = True)
			_,feed_dict['position'] = temp.sort(dim = 1)
			feed_dict['u_v'] = predict_dict['u_v'] # [batch(or num_sequence),ranker_embedding_len]
			feed_dict['i_v'] = predict_dict['i_v'] # [batch(or num_sequence),ranker_embedding_len]
			return feed_dict

class RerankSeqModel(RerankModel):
	reader='ImpressionSeqReader'
	runner='ImpressionRunner'
	extra_log_args = ['tuneranker']
		
	@staticmethod
	def parse_model_args(parser):
		parser.add_argument('--history_max', type=int, default=20,
							help='Maximum length of history.')
		parser.add_argument('--ranker_name', type=str, default='SASRec_test',
							help='Base ranker')
		parser.add_argument('--ranker_config_file', type=str, default='SASRec_test.yaml',
							help='Base ranker config file')
		parser.add_argument('--ranker_model_file', type=str, default='SASRec_test__MINDCTR__1__lr=0.0005__l2=0.0__emb_size=64__num_layers=3__num_heads=1.pt',
							help='Base ranker model file')
		parser.add_argument('--tuneranker', type=int, default=0,
							help='if 1, continue to train ranker')
		return ImpressionModel.parse_model_args(parser)
	
	def __init__(self, args, corpus):
		super().__init__(args, corpus)
		self.history_max = args.history_max
	
	class Dataset(ImpressionSeqModel.Dataset):		
		# Collate a batch according to the list of feed dicts
		def collate_batch(self, feed_dicts: List[dict]) -> dict: # feed_dicts are a batch of dicts
			feed_dict = super().collate_batch(feed_dicts)
			feed_dict['batch_size'] = len(feed_dicts)
			predict_dict = self.model.ranker(utils.batch_to_gpu(feed_dict, self.model.device)) # pos+pad+neg+pad
			feed_dict['scores'] = predict_dict['prediction'] # [batch(or num_sequence),n_candidate]
			pos_mask = torch.arange(0, self.model.train_max_pos_item, device = self.model.device).type_as(feed_dict['pos_num']).unsqueeze(0).expand(feed_dict['batch_size'], self.model.train_max_pos_item).lt(feed_dict['pos_num'].unsqueeze(1))
			neg_mask = torch.arange(0, self.model.train_max_neg_item, device = self.model.device).type_as(feed_dict['neg_num']).unsqueeze(0).expand(feed_dict['batch_size'], self.model.train_max_neg_item).lt(feed_dict['neg_num'].unsqueeze(1))
			all_mask = torch.cat([pos_mask, neg_mask],dim = 1)
			feed_dict['padding_mask'] = ~all_mask
			feed_dict['scores'] = torch.where(all_mask == 1, feed_dict['scores'],-np.inf * torch.ones_like(feed_dict['scores']))
			_,temp = feed_dict['scores'].sort(dim = 1, descending = True)
			_,feed_dict['position'] = temp.sort(dim = 1)
			feed_dict['u_v'] = predict_dict['u_v'] # [batch(or num_sequence),ranker_embedding_len]
			feed_dict['i_v'] = predict_dict['i_v'] # [batch(or num_sequence),ranker_embedding_len]

			#modeling user history, need all history item vector
			ranker = self.model.ranker
			if 'LightGCN' in self.model.ranker_name:
				all_his_its = ranker.encoder.embedding_dict['item_emb'][feed_dict['history_items'].to(self.model.device),:]
			else:
				all_his_its = ranker.i_embeddings(feed_dict['history_items'].to(self.model.device))
			feed_dict['his_v']=all_his_its
			return feed_dict