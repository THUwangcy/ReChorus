# -*- coding: UTF-8 -*-

import os
import logging
import numpy as np
from time import time
from tqdm import tqdm
from scipy.optimize import linprog

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader


def pct(dataset, predictions, max_topk, policy='Equal', personal=True, lambda_=0.5):
    """
    Personalized Calibration Target (PCT)
    :param dataset: the evaluation dataset
    :param predictions: relevance scores of candidate items for each instance in dataset
    :param max_topk: the length of recommendation lists
    :param policy: the policy to determine the overall target group exposure distribution \hat{q}
        a) Equal: each quality share the same exposure
        b) AvgEqual: the exposure proportional to the ratio in the item set
    :param personal: whether to use personalized calibration targets
        a) False: use the overall target group exposure distribution for each user
        b) True: solve personalized target distribution with linprog (PCT-Solver)
    :param lambda_: the tradeoff hyperparameter in reranking
    :return: the sorted idx of candidate items after reranking
    """
    class PCTDataset(Dataset):
        def __init__(self, inter_dataset, inter_pred):
            self.inter_dataset = inter_dataset
            self.inter_pred = inter_pred
            self.sort_idx = (-inter_pred).argsort()
            self.item2quality = inter_dataset.corpus.item2quality
            self.quality_level = inter_dataset.corpus.quality_level
            self.pos_weight = [1 / np.log2(pos + 2) for pos in range(max_topk)]

            """ PCT-Solver """
            self.target_exp = get_target_dist(inter_dataset, policy, personal)

        def __len__(self):
            return len(self.inter_dataset)

        def __getitem__(self, idx):  # rerank for a single user
            """ PCT-Reranker """
            selected = set()
            tmp_pred = self.inter_pred[idx].copy()
            rec_lst = np.ones_like(tmp_pred, dtype=np.int32) * -1  # final recommended items idx

            candidates = self.inter_dataset[idx]['item_id']
            quality_sign = [int(self.item2quality[item]) for item in candidates]

            user_target_exp = self.target_exp[idx] * np.sum(self.pos_weight)
            cur_exp = np.zeros(self.quality_level, dtype=np.float)

            # 1. First Iteration
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

            # 2. Second Iteration
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
    rerank_dataset = PCTDataset(dataset, predictions)
    dataloader = DataLoader(dataset=rerank_dataset, batch_size=8, num_workers=5)

    sort_idx = list()
    for rerank_res in tqdm(dataloader, leave=False, ncols=100, mininterval=1, desc='Rerank'):
        sort_idx.append(rerank_res.numpy())

    return np.concatenate(sort_idx, axis=0)


def get_target_dist(dataset, policy, personal=False):
    """
    Get the target group exposure distribution \hat{q}_u for each user.
    :param dataset: the evaluation dataset
    :param policy:
        a) Equal: each quality share the same exposure
        b) AvgEqual: the exposure proportional to the ratio in the item set
    :param personal:
        a) False: use the overall target group exposure distribution for each user
        b) True: solve personalized target distribution with linprog (PCT-Solver)
    :return: \hat{q}_u
    """
    item2quality = dataset.corpus.item2quality
    quality_level = dataset.corpus.quality_level
    item_quality = np.array(list(item2quality.values()))

    if policy in ['Equal']:  # equal
        target_dist = np.ones(quality_level) / quality_level
    elif policy in ['AvgEqual', 'test']:  # proportional to the ratio in the item set
        target_dist = np.zeros(quality_level)
        for q in range(quality_level):
            target_dist[q] = (item_quality == q).sum()
        target_dist /= len(item_quality)
    else:
        raise ValueError('Non-implemented policy. Choose from [Equal / AvgEqual].')

    if personal:  # solve personal target distribution (PCT-Solver)
        p_u = list()
        for i in range(len(dataset)):
            p_u.append(dataset.corpus.p_u[dataset[i]['user_id']])
        p_u = np.array(p_u)

        dataset_name, model_name = dataset.corpus.dataset, type(dataset.model).__name__
        seed = dataset.model.random_seed
        dist_path = os.path.join('../data/{}/pu-{}_{}-{}.npy'.format(dataset_name, model_name, seed, policy))
        target_dist = solve_per_target_dist(p_u, target_dist, path=dist_path)  # solve
    else:  # share global target distribution
        target_dist = [target_dist] * len(dataset)

    return target_dist  # [#test, #level]


def solve_per_target_dist(p_u, q_hat, path=None):
    """
    Core PCT-Solver.
    :param p_u: user historical interest distribution, [#user, #level]
    :param q_hat: target group exposure distribution, [#level]
    :param path: save and load path
    :return: personalized target group exposure distribution \hat{q}_u, [#user, #level]
    """
    if os.path.exists(path) and 'test' not in path:
        logging.info('Load personal target distribution from {}'.format(path))
        q_hat_u = np.load(path)
    else:
        logging.info('Solving personal target distribution... (may take a few minutes)')
        user_num = p_u.shape[0]
        p_u = p_u.transpose()  # [#level, #user]
        gradient = p_u.mean(1) - q_hat  # [#level]
        grad_direction = gradient / np.sqrt((gradient ** 2).sum())
        tile_grad = np.array([grad_direction] * user_num).transpose()  # [#level, #user]

        lim = tile_grad.copy()
        lim = np.where(tile_grad > 0, p_u / (tile_grad + 1e-10), lim)
        lim = np.where(tile_grad < 0, (p_u - 1) / (tile_grad + 1e-10), lim)
        lim = lim.min(0)  # [#user]

        A = tile_grad[:-1]  # [#level-1, #user]
        bounds = np.stack([np.zeros_like(lim), lim], axis=1)  # [#user, 2]
        weight = np.ones_like(lim)
        # b = (p_u - np.expand_dims(q_hat, 0).transpose()).sum(1)[:-1]  # [#level-1]
        # res = linprog(c=weight, A_eq=A, b_eq=b, bounds=bounds)

        chunk_size = 5000
        res_x = np.array([])
        for i in range(int(np.ceil(user_num / chunk_size))):
            begin, end = i * chunk_size, (i + 1) * chunk_size
            A_split, bounds_split = A[:, begin:end], bounds[begin:end, :]
            weight_split = weight[begin:end]
            b_split = (p_u[:, begin:end] - np.expand_dims(q_hat, 0).transpose()).sum(1)[:-1]
            res = linprog(c=weight_split, A_eq=A_split, b_eq=b_split, bounds=bounds_split)
            res_x = np.concatenate([res_x, res.x])

        q_hat_u = p_u - np.expand_dims(res_x, 0) * tile_grad  # [#level, #user]
        q_hat_u = q_hat_u.transpose()

        logging.info('Save personal target distribution to {}'.format(path))
        np.save(path, q_hat_u)

    return q_hat_u


"""
    Baselines
    1. TFROM (SIGIR'21)
    2. RegExp (SIGIR'22)
    3. Calibrated (Recsys'18)
"""
def tfrom(dataset, predictions, max_topk, policy='Equal', personal=False):
    # rank_weight = [1 for r in range(max_topk)]
    rank_weight = [1 / np.log2(r + 2) for r in range(max_topk)]  # weight of different ranking pos

    origin_order = (-predictions).argsort(axis=1)
    item2quality = dataset.corpus.item2quality
    quality_level = dataset.corpus.quality_level
    test_num = predictions.shape[0]

    target_dist = get_target_dist(dataset, policy, personal)

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


def reg_exp(dataset, predictions, max_topk, policy='Equal', personal=False, lambda_=1e-4):
    class MMRDataset(Dataset):
        def __init__(self, inter_dataset, inter_pred):
            self.inter_dataset = inter_dataset
            self.inter_pred = inter_pred
            self.item2quality = inter_dataset.corpus.item2quality
            self.quality_level = inter_dataset.corpus.quality_level

            self.pos_weight = [1 / np.log2(pos + 2) for pos in range(max_topk)]
            self.target_exp = get_target_dist(inter_dataset, policy, personal)

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
    dataloader = DataLoader(dataset=rerank_dataset, batch_size=81, num_workers=5)

    sort_idx = list()
    for rerank_res in tqdm(dataloader, leave=False, ncols=100, mininterval=1, desc='Rerank'):
        sort_idx.append(rerank_res.numpy())

    return np.concatenate(sort_idx, axis=0)


def calibrated(dataset, predictions, max_topk, lambda_=1e-4):
    class CalibratedDataset(Dataset):
        def __init__(self, inter_dataset, inter_pred):
            self.inter_dataset = inter_dataset
            self.inter_pred = inter_pred
            self.item2quality = inter_dataset.corpus.item2quality
            self.quality_level = inter_dataset.corpus.quality_level

            self.pos_weight = [1 / np.log2(pos + 2) for pos in range(max_topk)]

        def __len__(self):
            return len(self.inter_dataset)

        def __getitem__(self, idx):  # rerank for a single user
            selected = set()
            tmp_pred = self.inter_pred[idx].copy()
            rec_lst = np.zeros_like(tmp_pred, dtype=np.int32)

            candidates = self.inter_dataset[idx]['item_id']
            quality_sign = [int(self.item2quality[item]) for item in candidates]

            uid = self.inter_dataset[idx]['user_id']
            user_target_exp = np.array(self.inter_dataset.corpus.p_u[uid])
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
                    p = np.clip(user_target_exp, 1e-6, 1)
                    q = np.clip(assume_exp, 1e-6, 1)
                    kl = np.where(user_target_exp != 0, p * np.log(p / q), 0).sum()
                    score = lambda_ * tmp_pred[i] - (1 - lambda_) * kl
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
    rerank_dataset = CalibratedDataset(dataset, predictions)
    dataloader = DataLoader(dataset=rerank_dataset, batch_size=10, num_workers=10)

    sort_idx = list()
    for rerank_res in tqdm(dataloader, leave=False, ncols=100, mininterval=1, desc='Rerank'):
        sort_idx.append(rerank_res.numpy())

    return np.concatenate(sort_idx, axis=0)
