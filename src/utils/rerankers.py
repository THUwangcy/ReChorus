# -*- coding: UTF-8 -*-

import os
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader


def naive_boost(dataset, predictions, coef=0.1):
    # hqi_set = dataset.corpus.HQI_set
    quality_dict = dataset.corpus.item_meta_df['item_score3'].values
    tmp_pred = predictions.copy()
    for i in tqdm(range(len(dataset)), leave=False, ncols=100, mininterval=1, desc='Rerank'):
        candidates = dataset[i]['item_id']
        for idx in range(len(candidates)):
            quality = quality_dict[candidates[idx] - 1]
            tmp_pred[i][idx] += coef * quality
            # if candidates[idx] in hqi_set:
            #     tmp_pred[i][idx] += coef
    return (-tmp_pred).argsort(axis=1)


def tfrom(dataset, predictions, max_topk, policy='par'):
    origin_order = (-predictions).argsort(axis=1)
    item2quality = dataset.corpus.item_meta_df['i_quality'].values
    quality_level = int(item2quality.max() + 1)
    test_num = predictions.shape[0]

    if policy == 'cat':
        target_dist = np.zeros(quality_level)
        for q in range(quality_level):
            target_dist[q] = (item2quality == q).sum()
        target_dist /= len(item2quality)
        target_dist = [target_dist] * test_num
    elif policy == 'per':
        personal_dist_path = os.path.join(dataset.corpus.prefix, dataset.corpus.dataset, 'pu-rec-equal.npy')
        target_dist = np.load(personal_dist_path)
    else:
        target_dist = np.ones(quality_level) / quality_level
        target_dist = [target_dist] * test_num

    target_exp = np.zeros(quality_level)
    rank_weight = [1 / np.log2(r + 2) for r in range(max_topk)]
    cur_exp = np.zeros(quality_level, dtype=np.float)
    rec_lst = np.ones((test_num, max_topk), dtype=np.int) * -1
    tmp_pred = predictions.copy()

    for u in tqdm(range(test_num), leave=False, ncols=100, mininterval=1, desc='Rerank'):
        selected = set()
        target_exp += np.sum(rank_weight) * target_dist[u]
        for rank in range(max_topk):
            candidates = dataset[u]['item_id']
            for i in origin_order[u]:
                if i in selected:
                    continue
                quality = item2quality[candidates[i] - 1]
                if cur_exp[quality] + rank_weight[rank] <= target_exp[quality]:
                    rec_lst[u][rank] = i
                    cur_exp[quality] += rank_weight[rank]
                    selected.add(i)
                    tmp_pred[u][i] = -np.inf
                    break
        for rank in range(max_topk):
            candidates = dataset[u]['item_id']
            if rec_lst[u][rank] == -1:
                for i in origin_order[u]:
                    if i not in selected:
                        quality = item2quality[candidates[i] - 1]
                        rec_lst[u][rank] = i
                        cur_exp[quality] += rank_weight[rank]
                        selected.add(i)
                        tmp_pred[u][i] = -np.inf
                        break

    sort_idx = np.concatenate([rec_lst, (-tmp_pred).argsort(axis=1)[:, :-max_topk]], axis=1)
    return sort_idx


def max_marginal_rel(dataset, predictions, max_topk, policy='par', lambda_=0.5):
    class MMRDataset(Dataset):
        def __init__(self, inter_dataset, inter_pred):
            self.inter_dataset = inter_dataset
            self.inter_pred = inter_pred
            self.item2quality = inter_dataset.corpus.item_meta_df['i_quality'].values
            self.quality_level = int(self.item2quality.max() + 1)
            self.pos_weight = [1 / np.log2(pos + 2) for pos in range(max_topk)]
            if policy == 'cat':
                self.target_exp = np.zeros(self.quality_level)
                for q in range(self.quality_level):
                    self.target_exp[q] = (self.item2quality == q).sum()
                self.target_exp /= len(self.item2quality)
                self.target_exp = [self.target_exp] * len(self)
            elif policy == 'per':
                personal_dist_path = os.path.join(dataset.corpus.prefix, dataset.corpus.dataset, 'pu-test.npy')
                self.target_exp = np.load(personal_dist_path)
            else:
                self.target_exp = np.ones(self.quality_level) / self.quality_level
                # self.target_exp = np.array([0.4, 0.6])
                self.target_exp = [self.target_exp] * len(self)

        def __len__(self):
            return len(self.inter_dataset)

        def __getitem__(self, idx):
            selected = set()
            tmp_pred = self.inter_pred[idx].copy()
            rec_lst = np.zeros_like(tmp_pred, dtype=np.int32)

            candidates = self.inter_dataset[idx]['item_id']
            hqi_sign = [int(self.item2quality[item - 1]) for item in candidates]

            user_target_exp = self.target_exp[idx] * np.sum(self.pos_weight)
            cur_exp = np.zeros(self.quality_level, dtype=np.float)

            for pos in range(max_topk):
                # for each position
                max_score, max_idx = -np.inf, -1
                possible_exp = list()
                for q in range(self.quality_level):
                    tmp_exp = cur_exp.copy()
                    tmp_exp[q] += self.pos_weight[pos]
                    possible_exp.append(tmp_exp)

                for k in range(len(candidates)):
                    # for each candidate item
                    if k in selected:
                        continue
                    assume_exp = possible_exp[hqi_sign[k]]
                    disparity = ((np.sqrt(assume_exp) - np.sqrt(user_target_exp)) ** 2).sum() / 2
                    score = lambda_ * tmp_pred[k] - (1 - lambda_) * disparity
                    if score > max_score:
                        max_score = score
                        max_idx = k

                selected.add(max_idx)
                rec_lst[pos] = max_idx
                tmp_pred[max_idx] = -np.inf
                cur_exp[hqi_sign[max_idx]] += self.pos_weight[pos]

            rec_lst[max_topk:] = (-tmp_pred).argsort()[:-max_topk]
            return rec_lst

    torch.multiprocessing.set_sharing_strategy('file_system')
    rerank_dataset = MMRDataset(dataset, predictions)
    dataloader = DataLoader(dataset=rerank_dataset, batch_size=8, num_workers=5)

    sort_idx = list()
    for rerank_res in tqdm(dataloader, leave=False, ncols=100, mininterval=1, desc='Rerank'):
        sort_idx.append(rerank_res.numpy())

    return np.concatenate(sort_idx, axis=0)


def exp_slack(dataset, predictions, max_topk, policy='par'):
    class SlackDataset(Dataset):
        def __init__(self, inter_dataset, inter_pred):
            self.inter_dataset = inter_dataset
            self.inter_pred = inter_pred
            self.sort_idx = (-inter_pred).argsort()
            self.item2quality = inter_dataset.corpus.item_meta_df['i_quality'].values
            self.quality_level = int(self.item2quality.max() + 1)
            self.pos_weight = [1 / np.log2(pos + 2) for pos in range(max_topk)]
            if policy == 'cat':
                self.target_exp = np.zeros(self.quality_level)
                for q in range(self.quality_level):
                    self.target_exp[q] = (self.item2quality == q).sum()
                self.target_exp /= len(self.item2quality)
                self.target_exp = [self.target_exp] * len(self)
            elif policy == 'per':
                personal_dist_path = os.path.join(dataset.corpus.prefix, dataset.corpus.dataset, 'pu-directau-par.npy')
                self.target_exp = np.load(personal_dist_path)
            else:
                self.target_exp = np.ones(self.quality_level) / self.quality_level
                self.target_exp = [self.target_exp] * len(self)

        def __len__(self):
            return len(self.inter_dataset)

        def __getitem__(self, idx):
            selected = set()
            tmp_pred = self.inter_pred[idx].copy()
            rec_lst = np.ones_like(tmp_pred, dtype=np.int32) * -1

            candidates = self.inter_dataset[idx]['item_id']
            hqi_sign = [int(self.item2quality[item - 1]) for item in candidates]

            user_target_exp = self.target_exp[idx] * np.sum(self.pos_weight)
            cur_exp = np.zeros(self.quality_level, dtype=np.float)

            # for each position
            for pos in range(max_topk):

                # 1. determine the possible quality to be placed
                possible_quality = set()
                for q in range(self.quality_level):
                    if cur_exp[q] + self.pos_weight[pos] <= user_target_exp[q]:
                        possible_quality.add(q)
                if len(possible_quality) == 0:
                    quality_score = np.ones(self.quality_level) * -1
                    find_flag = 0
                    for k in self.sort_idx[idx]:
                        q = hqi_sign[k]
                        if candidates[k] not in selected and quality_score[q] == -1:
                            assume_exp = cur_exp.copy()
                            assume_exp[q] += self.pos_weight[pos]
                            disparity = ((assume_exp - user_target_exp) ** 2).sum() / 2
                            quality_score[q] = 0.8 * (1. / (k + 1)) - 0.2 * disparity
                            find_flag += 1
                            if find_flag == self.quality_level:
                                break
                    possible_quality.add(np.argmax(quality_score))

                # 2. find the first item with the possible quality
                for k in self.sort_idx[idx]:
                    quality = hqi_sign[k]
                    if candidates[k] not in selected and quality in possible_quality:
                        cur_exp[quality] += self.pos_weight[pos]
                        selected.add(candidates[k])
                        rec_lst[pos] = k
                        tmp_pred[k] = -np.inf
                        break

            rec_lst[max_topk:] = (-tmp_pred).argsort()[:-max_topk]
            return rec_lst

    torch.multiprocessing.set_sharing_strategy('file_system')
    rerank_dataset = SlackDataset(dataset, predictions)
    dataloader = DataLoader(dataset=rerank_dataset, batch_size=8, num_workers=5)

    sort_idx = list()
    for rerank_res in tqdm(dataloader, leave=False, ncols=100, mininterval=1, desc='Rerank'):
        sort_idx.append(rerank_res.numpy())

    return np.concatenate(sort_idx, axis=0)