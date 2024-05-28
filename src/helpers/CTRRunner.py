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

import sklearn.metrics as sk_metrics

class CTRRunner(BaseRunner):

	@staticmethod
	def evaluate_method(predictions: np.ndarray,labels: np.ndarray, metrics: list) -> Dict[str, float]:
		"""
		:param predictions: An array of predictions for all samples 
		:param labels: An array of labels for all samples (0 or 1)
		:param metrics: metric string list
		:return: a result dict, the keys are metrics
		"""
		evaluations = dict()
		for metric in metrics:
			if metric == 'ACC':
				evaluations[metric] = ((predictions>0.5).astype(int)==labels.astype(int)).mean()
			elif metric == 'AUC':
				evaluations[metric] = sk_metrics.roc_auc_score(labels,predictions)
			elif metric == 'F1_SCORE':
				evaluations[metric] = sk_metrics.f1_score(labels,(predictions>0.5).astype(int))
			elif metric == 'LOG_LOSS':
				clip_predictions = np.clip(predictions, a_min=1e-7, a_max=1-1e-7)
				evaluations[metric] = -(np.log(clip_predictions)*labels+ np.log(1-clip_predictions)*(1-labels)).mean()
			else:
				raise ValueError('Undefined evaluation metric: {}.'.format(metric))
		return evaluations

	def __init__(self, args):
		super().__init__(args)
		self.main_metric = self.metrics[0] if not len(args.main_metric) else self.main_metric
	
	def evaluate(self, dataset: BaseModel.Dataset, topks: list, metrics: list) -> Dict[str, float]:
		"""
		Evaluate the results for an input dataset.
		:return: result dict (key: metric)
		"""
		predictions, labels = self.predict(dataset)
		return self.evaluate_method(predictions, labels, metrics)

	def predict(self, dataset: BaseModel.Dataset, save_prediction: bool = False) -> np.ndarray:
		"""
		The returned prediction is a 1D-array corresponding to all samples,
		and ground truth labels are binary.
		"""
		dataset.model.eval()
		dataset.model.phase = 'eval'
		predictions, labels = list(), list()
		dl = DataLoader(dataset, batch_size=self.eval_batch_size, shuffle=False, num_workers=self.num_workers,
						collate_fn=dataset.collate_batch, pin_memory=self.pin_memory)
		for batch in tqdm(dl, leave=False, ncols=100, mininterval=1, desc='Predict'):
			if hasattr(dataset.model,'inference'):
				out_dict = dataset.model.inference(utils.batch_to_gpu(batch, dataset.model.device))
				prediction, label = out_dict['prediction'], out_dict['label']
			else:
				out_dict = dataset.model(utils.batch_to_gpu(batch, dataset.model.device))
				prediction, label = out_dict['prediction'], out_dict['label']
			predictions.extend(prediction.cpu().data.numpy())
			labels.extend(label.cpu().data.numpy())
		predictions = np.array(predictions)
		labels = np.array(labels)

		return predictions, labels
