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
                            help='Reranking method: Boost, Calibrated, RegExp, TFROM, PCT')
        parser.add_argument('--exp_policy', type=str, default='par',
                            help='Target exposure policy: par, cat')
        parser.add_argument('--personal', type=int, default=0,
                            help='Whether to solve personalized target exposure')
        parser.add_argument('--coef', type=float, default=0.1,
                            help='Coefficient of boosting')
        parser.add_argument('--lambda_', type=float, default=0.5,
                            help='Lambda in RegExp and PCT algorithm')
        return BaseRunner.parse_runner_args(parser)

    @staticmethod
    def get_exp_dist(origin_rec_item, item2quality, quality_level):
        """
        Calculate exposure distributions of different users.
        :return: user-specific original exposure distribution, [#user, #level]
        """
        logging.info('Calculating original exposure distribution...')
        q_h = list()
        for u in range(origin_rec_item.shape[0]):
            dist = [0] * quality_level
            for rank, item in enumerate(origin_rec_item[u]):
                quality = item2quality[item]
                delta = 1 / np.log2(rank + 2)
                dist[quality] += delta
            dist = dist / np.sum(dist)
            q_h.append(dist)
        return np.array(q_h)

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
        p_u = list()
        item2quality = dataset.corpus.item2quality
        for i in range(len(dataset)):
            candidates = dataset[i]['item_id']
            sort_lst = sort_idx[i][:max(topk)]
            rec_items.append([candidates[idx] for idx in sort_lst])
            rec_quality.append([item2quality[candidates[idx]] for idx in sort_lst])
            p_u.append(dataset.corpus.p_u[dataset[i]['user_id']])
        p_u = np.array(p_u)

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
                elif metric == 'COV':
                    # evaluations[key] = coverage_(rec_k_items, all_item)
                    eval_key = '{}_m@{}'.format(metric, k)
                    evaluations[eval_key] = coverage_(rec_k_items, hqi_group)
                elif metric == 'EXP':
                    eval_key = '{}_m@{}'.format(metric, k)
                    evaluations[eval_key] = ratio_(rec_k_items, hqi_group)
                elif metric == 'KL':
                    exp_dist = GuideRunner.get_exp_dist(rec_k_items, item2quality, dataset.corpus.quality_level)
                    p_u = np.clip(p_u, 1e-6, 1)
                    exp_dist = np.clip(exp_dist, 1e-6, 1)
                    kl = (p_u * np.log(p_u / exp_dist)).sum(1).mean()
                    evaluations[key] = kl
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
                return boosting(dataset, predictions, coef=self.coef)
            if self.rerank == 'RegExp':
                return reg_exp(dataset, predictions, max(self.topk), self.exp_policy, self.personal, self.lambda_)
            if self.rerank == 'TFROM':
                return tfrom(dataset, predictions, max(self.topk), self.exp_policy, self.personal)
            if self.rerank == 'PCT':
                return pct(dataset, predictions, max(self.topk), self.exp_policy, self.personal, self.lambda_)
            if self.rerank == 'Calibrated':
                return calibrated(dataset, predictions, max(self.topk), self.lambda_)
        return (-predictions).argsort(axis=1)
