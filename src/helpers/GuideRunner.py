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
                            help='Reranking method: boost, mmr')
        parser.add_argument('--exp_policy', type=str, default='par',
                            help='Target exposure policy: cat, int, per, par')
        parser.add_argument('--coef', type=float, default=0.1,
                            help='Coefficient of boosting')
        parser.add_argument('--lambda_', type=float, default=0.5,
                            help='Lambda in MMR algorithm')
        return BaseRunner.parse_runner_args(parser)

    def evaluate_metrics(self, dataset: BaseModel.Dataset, sort_idx: np.ndarray, topk: list, metrics: list,
                         eval_list: list) -> Dict[str, float]:
        """
        :param sort_idx: (-1, n_candidates) shape, argsort of the prediction result in descending order
        :param topk: top-K value list
        :param metrics: metric string list
        :param eval_list: name of specific sets list
        :return: a result dict, the keys are metric@topk
        """
        evaluations = dict()
        target_items = dataset.data['item_id']
        all_item = dataset.corpus.item_set

        gt_rank = np.argwhere(sort_idx == 0)[:, 1] + 1

        # get top rec_items
        rec_items = list()
        for i in range(len(dataset)):
            candidates = dataset[i]['item_id']
            sort_lst = sort_idx[i][:max(topk)]
            rec_items.append([candidates[idx] for idx in sort_lst])

        item_quality = np.array([dataset.corpus.item2quality[i] for i in target_items])

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
                    group_ndcg = list()
                    for quality in range(int(item_quality.max())):
                        group_ndcg.append(ndcg_lst[item_quality == quality].mean())
                    eval_key = '{}_CV@{}'.format(metric, k)
                    evaluations[eval_key] = np.std(group_ndcg) / np.mean(group_ndcg)
                    # for set_name in eval_list:
                    #     group = eval('dataset.corpus.{0}_set'.format(set_name))
                    #     eval_key = '{}_{}@{}'.format(metric, set_name, k)
                    #     evaluations[eval_key] = ndcg_(hit, gt_rank, target_items, group)
                    #     residual_group = all_item - group
                    #     eval_key = '{}_non{}@{}'.format(metric, set_name, k)
                    #     evaluations[eval_key] = ndcg_(hit, gt_rank, target_items, residual_group)
                elif metric == 'COV':
                    # evaluations[key] = coverage_(rec_k_items, all_item, len(all_item))
                    for set_name in eval_list:
                        group = eval('dataset.corpus.{0}_set'.format(set_name))
                        eval_key = '{}_{}@{}'.format(metric, set_name, k)
                        evaluations[eval_key] = coverage_(rec_k_items, group, len(all_item))
                elif metric == 'RATIO':
                    # evaluations[key] = ratio_(rec_k_items, all_item)  100%
                    for set_name in eval_list:
                        group = eval('dataset.corpus.{0}_set'.format(set_name))
                        eval_key = '{}_{}@{}'.format(metric, set_name, k)
                        evaluations[eval_key] = ratio_(rec_k_items, group)
                elif metric == 'PEN':
                    for set_name in eval_list:
                        group = eval('dataset.corpus.{0}_set'.format(set_name))
                        eval_key = '{}_{}@{}'.format(metric, set_name, k)
                        hqi_mask = np.isin(rec_k_items, list(group))
                        evaluations[eval_key] = (hqi_mask.sum(1) > 0).mean()
                elif metric == 'GINI':
                    evaluations[key] = gini_index_(rec_k_items, all_item)
                    for set_name in eval_list:
                        group = eval('dataset.corpus.{0}_set'.format(set_name))
                        eval_key = '{}_{}@{}'.format(metric, set_name, k)
                        evaluations[eval_key] = gini_index_(rec_k_items, group)
                else:
                    raise ValueError('Undefined evaluation metric: {}.'.format(metric))

        return evaluations

    def __init__(self, args):
        super().__init__(args)
        self.rerank = args.rerank
        self.exp_policy = args.exp_policy
        self.coef = args.coef
        self.lambda_ = args.lambda_

    def evaluate(self, dataset: BaseModel.Dataset, topks: list, metrics: list) -> Dict[str, float]:
        """
        Evaluate the results for an input dataset.
        :return: result dict (key: metric@k)
        """
        predictions = self.predict(dataset)
        sort_idx = self.pred_sort(dataset, predictions)
        return self.evaluate_metrics(dataset, sort_idx, topks, metrics, ['HQI'])

    def pred_sort(self, dataset: BaseModel.Dataset, predictions: np.ndarray):
        if dataset.phase == 'test':
            if self.rerank == 'boost':
                return naive_boost(dataset, predictions, coef=self.coef)
            if self.rerank == 'mmr':
                return max_marginal_rel(dataset, predictions, max(self.topk),
                                        policy=self.exp_policy, lambda_=self.lambda_)
        return (-predictions).argsort(axis=1)
