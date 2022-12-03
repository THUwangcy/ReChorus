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
from utils.metrics import *
from utils.rerankers import *
from models.BaseModel import BaseModel
from helpers.BaseRunner import BaseRunner


class GuideRunner(BaseRunner):
    @staticmethod
    def parse_runner_args(parser):
        parser.add_argument('--rerank', type=str, default=None,
                            help='Reranking method: Boost, RegExp, TFROM, PER')
        parser.add_argument('--exp_policy', type=str, default='par',
                            help='Target exposure policy: par, cat')
        parser.add_argument('--personal', type=int, default=0,
                            help='Whether to solve personalized target exposure')
        parser.add_argument('--coef', type=float, default=0.1,
                            help='Coefficient of boosting')
        parser.add_argument('--lambda_', type=float, default=0.5,
                            help='Lambda in RegExp and PER algorithm')
        return BaseRunner.parse_runner_args(parser)

    @staticmethod
    def evaluate_metrics(dataset: BaseModel.Dataset, sort_idx: np.ndarray,
                         topk: list, metrics: list) -> Dict[str, float]:
        """
        :param dataset: evaluation dataset class object
        :param sort_idx: (-1, n_candidates) shape, argsort of the prediction result in descending order
        :param topk: top-K value list
        :param metrics: metric string list
        :return: a result dict, the keys are metric@topk
        """
        # get top rec_items
        rec_items = list()
        rec_quality = list()
        item2quality = dataset.corpus.item2quality
        for i in range(len(dataset)):
            candidates = dataset[i]['item_id']
            sort_lst = sort_idx[i][:max(topk)]
            rec_items.append([candidates[idx] for idx in sort_lst])
            rec_quality.append([item2quality[candidates[idx]] for idx in sort_lst])

        evaluations = dict()
        target_items = dataset.data['item_id']
        target_quality = np.array([item2quality[i] for i in target_items])
        all_item = dataset.corpus.item_set
        hqi_group = dataset.corpus.HQI_set

        gt_rank = np.argwhere(sort_idx == 0)[:, 1] + 1

        for k in topk:
            hit = (gt_rank <= k)
            rec_k_items = np.array(rec_items)[:, :k]
            for metric in metrics:
                key = '{}@{}'.format(metric, k)
                if metric == 'HR':
                    evaluations[key] = hit.mean()
                elif metric == 'NDCG':
                    ndcg_lst = ndcg_(hit, gt_rank, target_items, all_item)
                    evaluations[key] = ndcg_lst.mean()

                    eval_key = 'CV@{}'.format(k)
                    group_ndcg = list()
                    for quality in range(int(target_quality.max()) + 1):
                        group_ndcg.append(ndcg_lst[target_quality == quality].mean())
                    evaluations[eval_key] = np.std(group_ndcg) / np.mean(group_ndcg)

                    eval_key = 'u-CV@{}'.format(k)
                    evaluations[eval_key] = np.std(ndcg_lst) / np.mean(ndcg_lst)

                    eval_key = '{}_HQI@{}'.format(metric, k)
                    evaluations[eval_key] = ndcg_(hit, gt_rank, target_items, hqi_group).mean()
                    residual_group = all_item - hqi_group
                    eval_key = '{}_nonHQI@{}'.format(metric, k)
                    evaluations[eval_key] = ndcg_(hit, gt_rank, target_items, residual_group).mean()
                elif metric == 'COV':
                    # evaluations[key] = coverage_(rec_k_items, all_item)
                    eval_key = '{}_HQI@{}'.format(metric, k)
                    evaluations[eval_key] = coverage_(rec_k_items, hqi_group)
                elif metric == 'RATIO':
                    eval_key = '{}_HQI@{}'.format(metric, k)
                    evaluations[eval_key] = ratio_(rec_k_items, hqi_group)
                elif metric == 'PEN':
                    eval_key = '{}_HQI@{}'.format(metric, k)
                    hqi_mask = np.isin(rec_k_items, list(hqi_group))
                    evaluations[eval_key] = (hqi_mask.sum(1) > 0).mean()
                elif metric == 'QS':
                    evaluations[key] = np.array(rec_quality)[:, :k].mean()
                else:
                    raise ValueError('Undefined evaluation metric: {}.'.format(metric))

        return evaluations

    def __init__(self, args):
        super().__init__(args)
        self.rerank = args.rerank
        self.exp_policy = args.exp_policy
        self.personal = args.personal
        self.coef = args.coef
        self.lambda_ = args.lambda_

    def evaluate(self, dataset: BaseModel.Dataset, topks: list, metrics: list) -> Dict[str, float]:
        """
        Evaluate the results for an input dataset.
        :return: result dict (key: metric@k)
        """
        predictions = self.predict(dataset)
        sort_idx = self.pred_sort(dataset, predictions)
        return self.evaluate_metrics(dataset, sort_idx, topks, metrics)

    def pred_sort(self, dataset: BaseModel.Dataset, predictions: np.ndarray):
        if dataset.phase in ['dev', 'test']:
            if self.rerank == 'Boost':
                return naive_boost(dataset, predictions, coef=self.coef)
            if self.rerank == 'RegExp':
                return reg_exp(dataset, predictions, max(self.topk), self.exp_policy, self.personal, self.lambda_)
            if self.rerank == 'TFROM':
                return tfrom(dataset, predictions, max(self.topk), self.exp_policy, self.personal)
            if self.rerank == 'PER':
                return per(dataset, predictions, max(self.topk), self.exp_policy, self.personal, self.lambda_)
        return (-predictions).argsort(axis=1)
