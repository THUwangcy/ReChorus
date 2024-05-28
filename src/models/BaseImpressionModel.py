import torch
import numpy as np
import torch.nn.functional as F
from typing import List
import yaml
import copy

from models.BaseModel import *

class ImpressionModel(GeneralModel):
	'''
	Positive & negative sample can be customized in training & testing.
	
	Refer to _append_impression_info() in ImpressionReader.py
	and the data generating jupyter notebook in dataset/
	
	for how to customize the samples.
	'''
	reader='ImpressionReader'
	runner='ImpressionRunner'

	@staticmethod
	def parse_model_args(parser):
		parser.add_argument('--loss_n', type=str, default='BPR',
							help='loss name,BPR or listnet or softmaxCE or attention_rank. list-level BPR variations: adding hard/not adding hard; choosing the reweight position: [after, before, between, simple] regarding the log(softmax(x)) in BPR. (default is between) So it can be like this: BPRhardafter, or BPR (default, just indicates not hard and between)')
		parser.add_argument('--train_max_pos_item', type=int, default=20,
						help='max positive item sample for training')
		parser.add_argument('--train_max_neg_item', type=int, default=20,
						help='max negative item sample for training')
		parser.add_argument('--test_max_pos_item', type=int, default=20,
						help='max positive item sample for evaluation')
		parser.add_argument('--test_max_neg_item', type=int, default=20,
						help='max negative item sample for evaluation')
		return GeneralModel.parse_model_args(parser)

	def __init__(self, args, corpus):
		super().__init__(args,corpus)
		self.loss_n = args.loss_n
		self.train_max_pos_item=args.train_max_pos_item
		self.test_max_pos_item=args.test_max_pos_item
		self.test_max_neg_item=args.test_max_neg_item
		self.train_max_neg_item=args.train_max_neg_item

	def loss(self, out_dict: dict, target=None):
		#multiple choices of list-wize ranking loss with optimization on multiple positive/negative samples
		prediction = out_dict['prediction'] # [batch_size, max_length_of_candidate_list]
		batch_size = prediction.size(0)
		cand_len = prediction.size(1)
		mask=torch.where(target==-1,target,torch.zeros_like(target))+1 # only non pad item is 1,shape like prediction
		test_have_neg = mask[:,self.train_max_pos_item] # if no neg 0, has neg 1

		if 'BPR' in self.loss_n: # Do not need to consider test_have_neg because the score_diff_mask has masked the places with no neg samples
			valid_mask = mask.unsqueeze(dim=-1) * mask.unsqueeze(dim=-1).transpose(-1,-2) # valid_mask[][i][j] is only 1 when mask[][i], mask[][j] are all 1
			pos_mask = (torch.arange(cand_len).unsqueeze(0).repeat(batch_size,1) < self.train_max_pos_item).to(self.device) # pos positions
			neg_mask = (torch.arange(cand_len).unsqueeze(0).repeat(batch_size,1) >= self.train_max_pos_item).to(self.device) # neg positions
			select_mask = pos_mask.unsqueeze(dim=-1) * neg_mask.unsqueeze(dim=-1).transpose(-1,-2) * valid_mask # get all valid mask in the two-dimensional matrix, select_mask[][i][j] is only 1 when mask[][i] is positive 1 (i<pos_num), mask[][j] is negative 1 (j>=pos_num)
			score_diff = prediction.unsqueeze(dim=-1) - prediction.unsqueeze(dim=-1).transpose(-1,-2) # batch * impression list * impression list
			score_diff_mask = score_diff * select_mask # score: pos - neg, only 1/4 matrix is not empty
			
            #Higher weights for high-score negative items
			neg_pred=torch.where(neg_mask*mask==1,prediction,-torch.tensor(float("Inf")).float().to(self.device))
			neg_softmax = (neg_pred - neg_pred.max()).softmax(dim=1) # [batch_size, max_length_of_candidate_list], only valid negative sample has softmax weights
			if 'hard' in self.loss_n: # Higher weights for lower-score positive items
				pos_pred=torch.where(pos_mask*mask==1,prediction,torch.tensor(float("Inf")).float().to(self.device))
				pos_softmax = (pos_pred.min() - pos_pred).softmax(dim=1) # [batch_size, max_length_of_candidate_list], only valid positive sample has softmax weights
			else: # Higher weights for high-score positive items
				pos_pred=torch.where(pos_mask*mask==1,prediction,-torch.tensor(float("Inf")).float().to(self.device))
				pos_softmax = (pos_pred - pos_pred.max()).softmax(dim=1)# [batch_size, max_length_of_candidate_list], only valid positive sample has softmax weights

			if 'after' in self.loss_n: # Do reweight across pos and neg samples after log-sigmoid. If only 1 pos and 1 neg, degrade to the BPR loss. The BPR loss is -log(softmax(x)). Softplus is log(1+e^x), which is equivalent of -log(softmax(-x)).
				loss = ((F.softplus(-score_diff_mask)*neg_softmax.unsqueeze(dim=1)).sum(dim=-1)*pos_softmax).sum(dim=-1)
				loss = loss.mean()
			elif 'before' in self.loss_n: # reweight before log-sigmoid
				loss = F.softplus(-(score_diff_mask*neg_softmax.unsqueeze(dim=1)).sum(dim=-1)*pos_softmax).sum(dim=-1)
				loss = loss.mean()
			elif 'simple' in self.loss_n: # Do not reweight, directly get the BPR of each pair
				loss = ((F.softplus(-score_diff_mask)*select_mask).sum(dim=-1)).sum(dim=-1)
			else: # reweight between log-sigmoid
				score_diff_mask = torch.where(select_mask==1, score_diff_mask, -torch.tensor(float("Inf")).float().to(self.device))
				loss = -((score_diff_mask.sigmoid()*neg_softmax.unsqueeze(dim=1)).sum(dim=-1)*pos_softmax).sum(dim=-1).log()
				loss = loss.mean()
			return loss

		elif self.loss_n=='listnet': #assume that the click probability is softmax(label) for pos and neg items (neg is not 0), and perform softmax cross entropy
			target=torch.where(target!=-1,target.float(),-torch.tensor(float("Inf")).float().to(self.device))

			target_softmax = (target-target.max()).softmax(dim=1)
			prediction_softmax = (prediction-prediction.max()).softmax(dim=1)
			prediction_softmax=torch.where(mask==1,prediction_softmax,torch.ones_like(prediction_softmax)) #make sure that paddings are 1, so that after log they are 0

			loss = -( target_softmax * prediction_softmax.log() ).sum(dim=1)
			loss = loss*test_have_neg/test_have_neg.sum()*len(test_have_neg)
			loss = loss.mean()
			return loss

		elif self.loss_n=='softmaxCE':#assume that the click probility is 1/k for k pos items, and zero for neg items, and perform softmax cross entropy
			pos_mask=torch.where(target==1,target,torch.zeros_like(target))
			pos_length=pos_mask.sum(axis=1)
			prediction=torch.where(mask==1,prediction,-torch.tensor(float("Inf")).float().to(self.device))
			pre_softmax = (prediction - prediction.max(dim=1, keepdim=True)[0]).softmax(dim=1)  # [batch_size, max_length_of_candidate_list]
			target_pre = pre_softmax[:, :self.train_max_pos_item]  # [batch_size, max_length_of_positive_samples]
			target_pre = torch.where(mask[:,:self.train_max_pos_item]==1,target_pre,torch.ones_like(target_pre))
			loss = -(target_pre).log().sum(axis=1).div(pos_length)

			loss = loss*test_have_neg/test_have_neg.sum()*len(test_have_neg)
			loss = loss.mean()
			return loss

		elif self.loss_n=='attention_rank': #softmax CE, but add more punishment for neg samples by adding the (1-label_i)log(1-predict_i) term
			target=torch.where(target!=-1,target.float(),-torch.tensor(float("Inf")).float().to(self.device))
			target_softmax = (target-target.max()).softmax(dim=1)

			prediction=torch.where(mask==1,prediction,-torch.tensor(float("Inf")).float().to(self.device))
			prediction_softmax = (prediction-prediction.max()).softmax(dim=1)

			prediction_softmax1 = torch.where(mask==1,prediction_softmax,torch.ones_like(prediction_softmax))#make sure that paddings are 1, so that after log they are 0
			loss_1 = -(target_softmax*prediction_softmax1.log()).sum(dim=1)

			prediction_softmax2 = torch.where(mask==1,prediction_softmax,torch.zeros_like(prediction_softmax))#make sure that paddings are 0, so that after log 1-they are 0
			prediction_softmax2 = torch.where(prediction_softmax2!=1,prediction_softmax2,torch.zeros_like(prediction_softmax2))#make sure that no 1 in prediction_softmax2, because it only happens when only 1 sample, so its loss must be 0
			loss_2 = -((1-target_softmax)*(1-prediction_softmax2).log()).sum(dim=1)

			loss = loss_1+loss_2
			loss = loss*test_have_neg/test_have_neg.sum()*len(test_have_neg)
			loss = loss.mean()
			return loss
		else:
			raise ValueError('Undefined loss function: {}'.format(self.loss_n))
		"""elif self.loss_n=='pointwiseCE':
			sample_length=mask.sum(axis=1)
			prediction = torch.sigmoid(out_dict['prediction'])
			loss = F.binary_cross_entropy(prediction,target.float(),reduction='none')
			loss = loss.mul(mask)
			return loss.sum(axis=1).div(sample_length).mean()

		elif self.loss_n=='sampled_softmax': 
			'''
			Reference:
				On the effectiveness of sampled softmax loss for item recommendation. Wu et al. 2022. Arxiv.
			'''
			pos_mask=torch.where(target==1,target,torch.zeros_like(target))
			relative_exp = (torch.exp(prediction*pos_mask)*pos_mask).sum(dim=-1) / (torch.exp(prediction*mask)*mask).sum(dim=-1)
			loss = -relative_exp.log()
			loss = loss.mean()
			return loss

		elif self.loss_n=='probCE':
			sample_length=mask.sum(axis=1)
			loss = F.binary_cross_entropy(prediction*mask,target.float(),reduction='none')
			loss = loss.mul(mask)
			# return loss.sum(axis=1).div(sample_length).mean()
			return loss.sum(axis=1).mean()"""


	class Dataset(GeneralModel.Dataset):
		def __init__(self, model, corpus, phase: str): # specify the max length of positive and negative samples
			super().__init__(model,corpus,phase)
			if self.phase=='train':
				self.pos_len=self.model.train_max_pos_item
				self.neg_len=self.model.train_max_neg_item
			else:
				self.pos_len=self.model.test_max_pos_item
				self.neg_len=self.model.test_max_neg_item

		def _get_feed_dict(self, index): # get feed dict with postive and negative samples and their actual length
			user_id, target_item = self.data['user_id'][index], self.data['pos_items'][index]
			if self.phase != 'train' and self.model.test_all:
				neg_items = np.arange(1, self.corpus.n_items)
			#if self.phase != 'train': # test negative sampling
				#neg_items = np.random.randint(1, self.corpus.n_items, size=20)
			else: # mostly this situation, customizing the neg items in evaluation
				neg_items = self.data['neg_items'][index]

			feed_dict = {
				'user_id': user_id,
				'pos_items': np.array(target_item[:self.pos_len]),
				'neg_items': np.array(neg_items[:self.neg_len]),
				'pos_num': min(self.data['pos_num'][index],self.pos_len),
				'neg_num': min(self.data['neg_num'][index],self.neg_len)
			}
			return feed_dict
		
		# Collate a batch according to the list of feed dicts
		def collate_batch(self, feed_dicts: List[dict]):
			feed_dict = super().collate_batch(feed_dicts)
			assert 'pos_items' in feed_dict and 'neg_items' in feed_dict

			pos_items = feed_dict['pos_items']
			if pos_items.shape[-1] < self.pos_len: # padding positive items
				pos_items = torch.cat((pos_items, torch.zeros(pos_items.shape[0],self.pos_len-pos_items.shape[-1])),dim=-1)
			neg_items = feed_dict['neg_items']
			if neg_items.shape[-1] < self.neg_len: # padding negative items
				neg_items = torch.cat((neg_items, torch.zeros(neg_items.shape[0],self.neg_len-neg_items.shape[-1])),dim=-1)
			feed_dict['item_id'] = torch.cat((pos_items,neg_items),dim=-1).long()
			feed_dict.pop('pos_items')
			feed_dict.pop('neg_items')
			return feed_dict
		
		def actions_before_epoch(self): 
			# Have to define it in order to use the pre-defined negative items for training.
			# Or else the negative sampling function of general model dataset will be called.
			# training set negative sampling
			'''neg_items = np.random.randint(1, self.corpus.n_items, size=(len(self), 20))
			for i, u in enumerate(self.data['user_id']):
				clicked_set = self.corpus.train_clicked_set[u]  # neg items are possible to appear in dev/test set
				# clicked_set = self.corpus.clicked_set[u]  # neg items will not include dev/test set
				for j in range(self.model.num_neg):
					while neg_items[i][j] in clicked_set:
						neg_items[i][j] = np.random.randint(1, self.corpus.n_items)
			self.data['neg_items'] = neg_items'''
			pass

class ImpressionSeqModel(ImpressionModel):
	reader='ImpressionSeqReader'
	runner='ImpressionRunner'
	
	@staticmethod
	def parse_model_args(parser):
		parser.add_argument('--history_max', type=int, default=20,
							help='Maximum length of history.')
		return ImpressionModel.parse_model_args(parser)
	
	def __init__(self, args, corpus):
		super().__init__(args, corpus)
		self.history_max = args.history_max

	class Dataset(SequentialModel.Dataset):
		def __init__(self, model, corpus, phase):
			super().__init__(model, corpus, phase)
			if self.phase=='train':
				self.pos_len=self.model.train_max_pos_item
				self.neg_len=self.model.train_max_neg_item
			else:
				self.pos_len=self.model.test_max_pos_item
				self.neg_len=self.model.test_max_neg_item
		
		def _get_feed_dict(self, index):
			feed_dict = ImpressionModel.Dataset._get_feed_dict(self,index)
			
			pos = self.data['position'][index]
			neg_pos = self.data['neg_position'][index]
			user_seq = self.corpus.user_his[feed_dict['user_id']]['pos'][:pos]#positive
			neg_user_seq = self.corpus.user_his[feed_dict['user_id']]['neg'][:neg_pos]#negative
			if self.model.history_max > 0:
				user_seq = user_seq[-self.model.history_max:]
				neg_user_seq = neg_user_seq[-self.model.history_max:]
			feed_dict['history_items'] = np.array([x[0] for x in user_seq])
			feed_dict['neg_history_items'] = np.array([x[0] for x in neg_user_seq])
			feed_dict['history_times'] = np.array([x[1] for x in user_seq])
			feed_dict['neg_history_times'] = np.array([x[1] for x in neg_user_seq])
			feed_dict['lengths'] = len(feed_dict['history_items'])
			feed_dict['neg_lengths'] = len(feed_dict['neg_history_items'])
			return feed_dict
		
		# Collate a batch according to the list of feed dicts
		def collate_batch(self, feed_dicts: List[dict]):
			feed_dict = super().collate_batch(feed_dicts)
			assert 'pos_items' in feed_dict and 'neg_items' in feed_dict

			pos_items = feed_dict['pos_items']
			if pos_items.shape[-1] < self.pos_len: # padding positive items
				pos_items = torch.cat((pos_items, torch.zeros(pos_items.shape[0],self.pos_len-pos_items.shape[-1])),dim=-1)
			neg_items = feed_dict['neg_items']
			if neg_items.shape[-1] < self.neg_len: # padding negative items
				neg_items = torch.cat((neg_items, torch.zeros(neg_items.shape[0],self.neg_len-neg_items.shape[-1])),dim=-1)
			feed_dict['item_id'] = torch.cat((pos_items,neg_items),dim=-1).long()
			feed_dict.pop('pos_items')
			feed_dict.pop('neg_items')
			
			feed_dict['history_items'] = feed_dict['history_items'].long()
			feed_dict['neg_history_items'] = feed_dict['neg_history_items'].long()
			return feed_dict
		
		def actions_before_epoch(self): 
			# training negatives, same as non-seq impression models
			ImpressionModel.Dataset.actions_before_epoch(self)
			pass