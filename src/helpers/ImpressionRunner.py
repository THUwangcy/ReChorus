# -*- coding: UTF-8 -*-

import os
import gc
import torch
import torch.nn as nn
import logging
import numpy as np
from time import time
from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import Dict, List

from utils import utils
from models.BaseModel import BaseModel
from helpers.BaseRunner import BaseRunner

def HR_at_k(labels, valid_num, k):
	indices = np.arange(labels.shape[1]) < valid_num[:, None]
	labels = labels * indices
	num_hits = np.sum(labels[:,:k], axis=1)

	# Hit rate at k: one positive before k then hitrate of this list is 1
	positive_num = np.sum(labels, axis=1)
	positive_num[positive_num == 0] = 1
	positive_num[positive_num > k] = k # when have more than k positives, attention
	
	hit_rate = num_hits / positive_num
	hit_rate[hit_rate > 0] = 1
	return hit_rate

def DCG_at_k(labels, valid_num, k):
	indices = np.arange(labels.shape[1]) < valid_num[:, None]
	labels = labels * indices
	labels = labels[:,:k]
	dcg = np.sum(labels / np.log2(np.arange(2, labels.shape[1] + 2)), axis=1)    
	return dcg

def NDCG_at_k(labels, valid_num, k):
	indices = np.arange(labels.shape[1]) < valid_num[:, None]
	labels = labels * indices
	dcg = DCG_at_k(labels, valid_num, k)  # DCG @ k of each row
	
	# ideal DCG
	sorted_labels = np.sort(labels, axis=1)[:, ::-1]
	ideal_dcg = DCG_at_k(sorted_labels, valid_num, k)
	
	ideal_dcg[ideal_dcg == 0] = 1
	ndcg = dcg / ideal_dcg
	
	return ndcg

def AP_at_k(labels, valid_num, k):
	indices = np.arange(labels.shape[1]) < valid_num[:, None]
	labels = labels * indices
	
	# Precision for each row
	num_positive_predictions = np.cumsum(labels, axis=1)
	num_positive_predictions[:, k:] = 0
	precision = num_positive_predictions / np.arange(1, labels.shape[1] + 1)
	
	positive_num = np.sum(labels, axis=1)
	positive_num[positive_num == 0] = 1
	positive_num[positive_num > k] = k # when have more than k positives, attention
	average_precision = np.sum(precision * labels, axis=1) / positive_num
	return average_precision

class ImpressionRunner(BaseRunner):
	@staticmethod
	def parse_runner_args(parser):
		return BaseRunner.parse_runner_args(parser)

	@staticmethod
	def evaluate_method(predictions: np.ndarray, topk: list, metrics: list, test_all: bool, neg_num, pos_num_max, pos_num = None, check_sort_idx = 0, test_num_neg = 0,ret_all = 0) -> Dict[str, float]:
		"""
		:param predictions: (-1, n_candidates) shape, when pos_num=None, the first column is the score for ground-truth item, if pos_num!=None, the 0:pos_num column is ground-truth. Also, pos_num:pos_num+neg_num is negative item
		:param topk: top-K value list
		:param metrics: metric string list
		:return: a result dict, the keys are metric@topk
		"""
		evaluations = dict()
		if test_all:
			pass
		else:
			if pos_num is None:
				pos_num=[1 for i in range(len(predictions))]
			#predictions: pos, -inf, neg, -inf

			#make sure that positive items will be ranked lower than neg items, when they have the same prediction values
			pos_mask = np.ones((predictions.shape[0],pos_num_max))
			rest_mask = np.zeros((predictions.shape[0],predictions.shape[1]-pos_num_max))
			a_mask = np.concatenate((pos_mask,rest_mask),axis=1)
			eps=1e-6
			predictions=predictions-eps*a_mask


			sort_idx = (-predictions).argsort(axis=1,kind='mergesort') # mergesort keeps the order
			if check_sort_idx==1:
				logging.info(str(sort_idx[:10]))
			
			neg_num_max = len(predictions[0])-pos_num_max
			pos_num_cliped = np.array(pos_num)
			pos_num_cliped[pos_num_cliped > pos_num_max] = pos_num_max
			neg_num_cliped = np.array(neg_num)
			neg_num_cliped[neg_num_cliped > neg_num_max] = neg_num_max
			whole_len = pos_num_cliped + neg_num_cliped

			labels = (np.arange(max(max(pos_num_cliped),pos_num_max)) < pos_num_cliped[:, None]).astype(int)
			labels = np.concatenate((labels,np.zeros_like(labels)),axis=1)
			labels = np.take_along_axis(labels, sort_idx, axis=1)

			for k in topk:
				ndcg = NDCG_at_k(labels, whole_len, k)
				if ret_all == 0: 
					evaluations['NDCG@{}'.format(k)] = ndcg.mean()
				else:
					evaluations['NDCG@{}'.format(k)] = ndcg
			
			for k in topk:
				map = AP_at_k(labels, whole_len, k)
				if ret_all == 0: 
					evaluations['MAP@{}'.format(k)] = map.mean()
				else:
					evaluations['MAP@{}'.format(k)] = map

			for k in topk:
				hr = HR_at_k(labels, whole_len, k)
				if ret_all == 0:
					evaluations['HR@{}'.format(k)] = hr.mean()
				else:
					evaluations['HR@{}'.format(k)] = hr
  
		return evaluations

	def evaluate(self, data: BaseModel.Dataset, topks: list, metrics: list, check_sort_idx = 0, all = 0) -> Dict[str, float]:
		"""
		Evaluate the results for an input dataset.
		:return: result dict (key: metric@k)
		"""
		predictions = self.predict(data)
		if data.model.test_all:
			rows, cols = list(), list()
			for i, u in enumerate(data.data['user_id']):
				clicked_items = [x[0] for x in data.corpus.user_his[u]]
				# clicked_items = [data.data['item_id'][i]]
				idx = list(np.ones_like(clicked_items) * i)
				rows.extend(idx)
				cols.extend(clicked_items)
			predictions[rows, cols] = -np.inf

		rows, cols = list(), list()
		mask = np.full_like(predictions, 0)
		if 'pos_num' not in data.data.keys():
			pos_num=[1 for i in range(len(predictions))]
		else:
			pos_num=data.data['pos_num']
		neg_num=data.data['neg_num']
		mp = data.model.test_max_pos_item
		mn = data.model.test_max_neg_item
		for i in range(len(data.data['neg_num'])):
			rows.extend([i for _ in range(min(pos_num[i], mp))])
			rows.extend([i for _ in range(min(neg_num[i], mn))])
			cols.extend([_ for _ in range(min(pos_num[i], mp))])
			cols.extend([_ for _ in range(mp, mp+min(neg_num[i], mn))])
		mask[rows, cols] = 1

		predictions = np.where(mask == 1, predictions, -np.inf)
		if 'pos_num' in data.data.keys():
			return self.evaluate_method(predictions, topks, metrics, data.model.test_all, data.data['neg_num'], mp, data.data['pos_num'], check_sort_idx, test_num_neg=data.neg_len, ret_all=all)
		else:
			return self.evaluate_method(predictions, topks, metrics, data.model.test_all, data.data['neg_num'], mp, check_sort_idx, test_num_neg=data.neg_len, ret_all=all)
	
	def fit(self, data: BaseModel.Dataset, epoch = -1) -> float:
		model = data.model
		if model.optimizer is None:
			model.optimizer = self._build_optimizer(model)
		data.actions_before_epoch()  # must sample before multi thread start

		model.train()
		loss_lst = list()
		dl = DataLoader(data, batch_size = self.batch_size, shuffle = True, num_workers = self.num_workers,
						collate_fn = data.collate_batch, pin_memory=self.pin_memory)
		for batch in tqdm(dl, leave = False, desc = 'Epoch {:<3}'.format(epoch), ncols = 100, mininterval = 1):
			batch = utils.batch_to_gpu(batch, model.device)
			model.optimizer.zero_grad()
			out_dict = model(batch)
			max_pos_num = model.train_max_pos_item
			pos_mask = 2*(torch.arange(max_pos_num)[None, :].to(model.device) < batch['pos_num'][:, None]).int()-1
			neg_mask = (torch.arange(out_dict['prediction'].size(1) - max_pos_num)[None, :].to(model.device) < batch['neg_num'][:, None]).int() - 1
			labels = torch.cat([pos_mask, neg_mask], dim = -1)
			loss = model.loss(out_dict, labels)
			if loss.isnan() or loss.isinf() or out_dict['prediction'].isnan().any() or out_dict['prediction'].isinf().any():
				logging.info("Loss is Nan. Stop training at %d."%(epoch + 1))
			loss.backward()
			model.optimizer.step()
			loss_lst.append(loss.detach().cpu().data.numpy())
		return np.mean(loss_lst).item()
