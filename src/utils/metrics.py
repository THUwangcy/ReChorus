import os
import logging
import torch
import numpy as np
import pandas as pd
from collections import Counter

def hr_(hit, target_items, item_set):
    idx = np.in1d(target_items, list(item_set))
    return hit[idx]

def ndcg_(hit, gt_rank, target_items, item_set):
    idx = np.in1d(target_items, list(item_set))
    ndcg = (hit[idx] / np.log2(gt_rank[idx] + 1))
    return ndcg

def coverage_(rec_items, item_set):
    return len(set(np.concatenate(rec_items)) & item_set) / len(item_set)

def ratio_(rec_items, item_set):
    items = np.concatenate(rec_items)
    idx = np.in1d(items, list(item_set))
    # return idx.sum() / len(items)
    test_num, k = rec_items.shape
    coef = [1 / np.log2(pos + 2) for pos in range(k)]
    coef = np.array([coef] * test_num)
    coef = coef / coef.sum(1, keepdims=True)
    coef = np.concatenate(coef)
    return (idx * coef).sum() / test_num

def gini_index_(rec_items, item_set):
    item_idx = np.in1d(rec_items.flatten(), list(item_set))
    new_rec_items = rec_items.flatten()[item_idx]
    n_items = len(item_set)

    item_count = dict(Counter(new_rec_items))
    sorted_count = np.array(sorted(item_count.values()))
    num_recommended_items = sorted_count.shape[0]
    total_num = len(new_rec_items)
    idx = np.arange(n_items - num_recommended_items + 1, n_items + 1)
    gini_index = np.sum((2 * idx - n_items - 1) * sorted_count) / total_num
    gini_index /= n_items
    return gini_index
