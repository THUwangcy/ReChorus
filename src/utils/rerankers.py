# -*- coding: UTF-8 -*-

import os
import logging
import numpy as np
from tqdm import tqdm
from scipy.optimize import linprog

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader


"""
    Get target exposure distribution for each user.
    1. policy
        a) par: each quality share the same exposure
        b) cat: the exposure proportional to the ratio in the item set
    2. personal
        a) False: use global target distribution for each user
        b) True: solve personal target distribution with linprog
"""
def get_target_dist(dataset, predictions, max_topk, policy, personal=False):
    item2quality = dataset.corpus.item2quality
    quality_level = dataset.corpus.quality_level
    item_quality = np.array(list(item2quality.values()))

    if policy == 'par':  # equal
        target_dist = np.ones(quality_level) / quality_level
    elif policy == 'cat':  # proportional to the ratio in the item set
        target_dist = np.zeros(quality_level)
        for q in range(quality_level):
            target_dist[q] = (item_quality == q).sum()
        target_dist /= len(item_quality)
    else:
        raise ValueError('Non-implemented policy. Choose from [par / cat].')

    if personal:  # solve personal target distribution
        origin_rec_items = list()
        origin_order = (-predictions).argsort(axis=1)[:, :max_topk]
        for i in range(len(dataset)):
            candidates = dataset[i]['item_id']
            sort_lst = origin_order[i]
            origin_rec_items.append([candidates[idx] for idx in sort_lst])
        origin_rec_items = np.array(origin_rec_items)

        q_h = get_original_dist(origin_rec_items, item2quality, quality_level)  # original dist

        dataset_name, model_name = dataset.corpus.dataset, type(dataset.model).__name__
        # model_name, policy = 'test', 'equal'
        dist_path = os.path.join('../data/{}/pu-{}-{}.npy'.format(dataset_name, model_name, policy))
        target_dist = solve_per_target_dist(q_h, target_dist, path=dist_path)  # solve
    else:  # share global target distribution
        target_dist = [target_dist] * len(dataset)

    return target_dist  # [#test, #level]


def get_original_dist(origin_rec_item, item2quality, quality_level):
    """
    Calculate original exposure distributions of different users.
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


def solve_per_target_dist(q_h, target_dist, path=None):
    """
    :param q_h: user original exposure distribution, [#user, #level]
    :param target_dist: target exposure distribution, [#level]
    :param path: save and load path
    :return: personal target exposure distribution, [#user, #level]
    """
    if os.path.exists(path) and 'test' not in path:
        logging.info('Load personal target distribution from {}'.format(path))
        p_u = np.load(path)
    else:
        logging.info('Solving personal target distribution... (may take a few minutes)')
        user_num = q_h.shape[0]
        q_h = q_h.transpose()  # [#level, #user]
        gradient = q_h.mean(1) - target_dist  # [#level]
        grad_direction = gradient / np.sqrt((gradient ** 2).sum())
        tile_grad = np.array([grad_direction] * user_num).transpose()  # [#level, #user]

        lim = tile_grad.copy()
        lim = np.where(tile_grad > 0, q_h / (tile_grad + 1e-10), lim)
        lim = np.where(tile_grad < 0, (q_h - 1) / (tile_grad + 1e-10), lim)
        lim = lim.min(0)  # [#user]

        A = tile_grad[:-1]  # [#level-1, #user]
        b = (q_h - np.expand_dims(target_dist, 0).transpose()).sum(1)[:-1]  # [#level-1]
        bounds = np.stack([np.zeros_like(lim), lim], axis=1)  # [#user, 2]
        weight = np.ones_like(lim)
        res = linprog(c=weight, A_eq=A, b_eq=b, bounds=bounds)

        p_u = q_h - np.expand_dims(res.x, 0) * tile_grad  # [#level, #user]
        p_u = p_u.transpose()

        logging.info('Save personal target distribution to {}'.format(path))
        np.save(path, p_u)

    return p_u


"""
    Reranking algorithms
    1. Boost
    2. TFROM (SIGIR'21)
    3. RegExp (SIGIR'22)
    4. PER (ours)
"""
def naive_boost(dataset, predictions, coef=0.1):
    if dataset.corpus.dataset in ['QK-article-1M']:
        item_meta_df = dataset.corpus.item_meta_df
        item2quality = dict(zip(item_meta_df['item_id'], item_meta_df['item_score3']))
    else:
        item2quality = dataset.item2quality
    tmp_pred = predictions.copy()
    for i in tqdm(range(len(dataset)), leave=False, ncols=100, mininterval=1, desc='Rerank'):
        candidates = dataset[i]['item_id']
        for idx in range(len(candidates)):
            quality = item2quality[candidates[idx]]
            tmp_pred[i][idx] += coef * quality
    return (-tmp_pred).argsort(axis=1)


def tfrom(dataset, predictions, max_topk, policy='par', personal=False):
    # rank_weight = [1 for r in range(max_topk)]
    rank_weight = [1 / np.log2(r + 2) for r in range(max_topk)]  # weight of different ranking pos

    origin_order = (-predictions).argsort(axis=1)
    item2quality = dataset.corpus.item2quality
    quality_level = dataset.corpus.quality_level
    test_num = predictions.shape[0]

    target_dist = get_target_dist(dataset, predictions, max_topk, policy, personal)

    target_exp = np.zeros(quality_level)  # target exposure up to now
    cur_exp = np.zeros(quality_level, dtype=np.float)  # current exposure up to now
    rec_lst = np.ones((test_num, max_topk), dtype=np.int) * -1
    tmp_pred = predictions.copy()

    for u in tqdm(range(test_num), leave=False, ncols=100, mininterval=1, desc='Rerank'):
        selected = set()
        target_exp += np.sum(rank_weight) * target_dist[u]  # update target exposure according to current user
        candidates = dataset[u]['item_id']
        for rank in range(max_topk):  # for each rec position
            for i in origin_order[u]:
                if i in selected:
                    continue
                quality = item2quality[candidates[i]]
                if cur_exp[quality] + rank_weight[rank] <= target_exp[quality]:  # not exceed the target exposure
                    rec_lst[u][rank] = i
                    cur_exp[quality] += rank_weight[rank]
                    selected.add(i)
                    tmp_pred[u][i] = -np.inf
                    break
        for rank in range(max_topk):  # fill still empty positions
            if rec_lst[u][rank] == -1:
                for i in origin_order[u]:
                    if i not in selected:  # directly select the top-unselected item
                        quality = item2quality[candidates[i]]
                        rec_lst[u][rank] = i
                        cur_exp[quality] += rank_weight[rank]
                        selected.add(i)
                        tmp_pred[u][i] = -np.inf
                        break

    sort_idx = np.concatenate([rec_lst, (-tmp_pred).argsort(axis=1)[:, :-max_topk]], axis=1)
    assert (sort_idx < 0).sum() == 0
    return sort_idx


def reg_exp(dataset, predictions, max_topk, policy='par', personal=False, lambda_=1e-4):
    class MMRDataset(Dataset):
        def __init__(self, inter_dataset, inter_pred):
            self.inter_dataset = inter_dataset
            self.inter_pred = inter_pred
            self.item2quality = inter_dataset.corpus.item2quality
            self.quality_level = inter_dataset.corpus.quality_level

            self.pos_weight = [1 / np.log2(pos + 2) for pos in range(max_topk)]
            self.target_exp = get_target_dist(inter_dataset, inter_pred, max_topk, policy, personal)

        def __len__(self):
            return len(self.inter_dataset)

        def __getitem__(self, idx):  # rerank for a single user
            selected = set()
            tmp_pred = self.inter_pred[idx].copy()
            rec_lst = np.zeros_like(tmp_pred, dtype=np.int32)

            candidates = self.inter_dataset[idx]['item_id']
            quality_sign = [int(self.item2quality[item]) for item in candidates]

            user_target_exp = self.target_exp[idx] * np.sum(self.pos_weight)
            cur_exp = np.zeros(self.quality_level, dtype=np.float)

            for rank in range(max_topk):  # for each rec position
                max_score, max_idx = -np.inf, -1
                possible_exp = list()
                for q in range(self.quality_level):
                    tmp_exp = cur_exp.copy()
                    tmp_exp[q] += self.pos_weight[rank]
                    possible_exp.append(tmp_exp)

                for i in range(len(candidates)):  # for each candidate item
                    if i in selected:
                        continue
                    assume_exp = possible_exp[quality_sign[i]]
                    disparity = ((np.sqrt(assume_exp) - np.sqrt(user_target_exp)) ** 2).sum() / 2
                    score = lambda_ * tmp_pred[i] - (1 - lambda_) * disparity
                    if score > max_score:
                        max_score = score
                        max_idx = i

                selected.add(max_idx)
                rec_lst[rank] = max_idx
                tmp_pred[max_idx] = -np.inf
                cur_exp[quality_sign[max_idx]] += self.pos_weight[rank]

            rec_lst[max_topk:] = (-tmp_pred).argsort()[:-max_topk]
            return rec_lst

    torch.multiprocessing.set_sharing_strategy('file_system')
    rerank_dataset = MMRDataset(dataset, predictions)
    dataloader = DataLoader(dataset=rerank_dataset, batch_size=8, num_workers=5)

    sort_idx = list()
    for rerank_res in tqdm(dataloader, leave=False, ncols=100, mininterval=1, desc='Rerank'):
        sort_idx.append(rerank_res.numpy())

    return np.concatenate(sort_idx, axis=0)


def per(dataset, predictions, max_topk, policy='par', personal=True, lambda_=0.5):
    class SlackDataset(Dataset):
        def __init__(self, inter_dataset, inter_pred):
            self.inter_dataset = inter_dataset
            self.inter_pred = inter_pred
            self.sort_idx = (-inter_pred).argsort()
            self.item2quality = inter_dataset.corpus.item2quality
            self.quality_level = inter_dataset.corpus.quality_level

            self.pos_weight = [1 / np.log2(pos + 2) for pos in range(max_topk)]
            self.target_exp = get_target_dist(inter_dataset, inter_pred, max_topk, policy, personal)

        def __len__(self):
            return len(self.inter_dataset)

        def __getitem__(self, idx):  # rerank for a single user
            selected = set()
            tmp_pred = self.inter_pred[idx].copy()
            rec_lst = np.ones_like(tmp_pred, dtype=np.int32) * -1  # final recommended items idx

            candidates = self.inter_dataset[idx]['item_id']
            quality_sign = [int(self.item2quality[item]) for item in candidates]

            user_target_exp = self.target_exp[idx] * np.sum(self.pos_weight)
            cur_exp = np.zeros(self.quality_level, dtype=np.float)

            # 1. fill in the positions with slack
            for pos in range(max_topk):  # for each position to be recommended
                for k in self.sort_idx[idx]:
                    quality = quality_sign[k]
                    assume_exp = cur_exp[quality] + self.pos_weight[pos]
                    if assume_exp <= user_target_exp[quality] and candidates[k] not in selected:
                        cur_exp[quality] = assume_exp
                        selected.add(candidates[k])
                        rec_lst[pos] = k
                        tmp_pred[k] = -np.inf
                        break

            # 2. fill in the blanks with MMR
            for pos in range(max_topk):
                if rec_lst[pos] == -1:  # all the quality exceed the target exposure
                    quality_score = np.ones(self.quality_level) * -1  # score of top-item for each quality
                    item_idx = np.ones(self.quality_level) * -1
                    for r, k in enumerate(self.sort_idx[idx]):
                        q = quality_sign[k]
                        if quality_score[q] == -1 and candidates[k] not in selected:
                            assume_exp = cur_exp.copy()
                            assume_exp[q] += self.pos_weight[pos]
                            disparity = ((assume_exp - user_target_exp) ** 2).sum() / 2
                            quality_score[q] = lambda_ * (1. / (r + 1)) - (1 - lambda_) * disparity
                            item_idx[q] = k
                            if (quality_score < 0).sum() == 0:
                                break
                    max_k = int(item_idx[np.argmax(quality_score)])
                    quality = quality_sign[max_k]
                    cur_exp[quality] += self.pos_weight[pos]
                    selected.add(candidates[max_k])
                    rec_lst[pos] = max_k
                    tmp_pred[max_k] = -np.inf

            rec_lst[max_topk:] = (-tmp_pred).argsort()[:-max_topk]
            return rec_lst

    torch.multiprocessing.set_sharing_strategy('file_system')
    rerank_dataset = SlackDataset(dataset, predictions)
    dataloader = DataLoader(dataset=rerank_dataset, batch_size=8, num_workers=5)

    sort_idx = list()
    for rerank_res in tqdm(dataloader, leave=False, ncols=100, mininterval=1, desc='Rerank'):
        sort_idx.append(rerank_res.numpy())

    return np.concatenate(sort_idx, axis=0)