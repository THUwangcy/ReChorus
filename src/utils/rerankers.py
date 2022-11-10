# -*- coding: UTF-8 -*-

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


def max_marginal_rel(dataset, predictions, max_topk, policy='cat', lambda_=0.5):
    class MMRDataset(Dataset):
        def __init__(self, inter_dataset, inter_pred):
            self.inter_dataset = inter_dataset
            self.inter_pred = inter_pred
            self.item2quality = inter_dataset.corpus.item_meta_df['i_quality'].values
            self.quality_level = int(self.item2quality.max() + 1)
            if policy == 'cat':
                self.target_exp = np.zeros(self.quality_level)
                for q in range(self.quality_level):
                    self.target_exp[q] = (self.item2quality == q).sum()
                self.target_exp /= len(self.item2quality)
            else:
                self.target_exp = np.ones(self.quality_level) / self.quality_level

        def __len__(self):
            return len(self.inter_dataset)

        def __getitem__(self, idx):
            selected = set()
            tmp_pred = self.inter_pred[idx].copy()
            rec_lst = np.zeros_like(tmp_pred, dtype=np.int32)

            candidates = self.inter_dataset[idx]['item_id']
            hqi_sign = [int(self.item2quality[item - 1]) for item in candidates]

            cur_exp = np.zeros(self.quality_level, dtype=np.float)
            norm_factor = 0

            for pos in range(max_topk):
                # for each position
                max_score, max_idx = -np.inf, -1
                rank_contri = 1 / np.log2(pos + 2)
                norm_factor += rank_contri
                possible_exp = list()
                for q in range(self.quality_level):
                    tmp_exp = cur_exp.copy()
                    tmp_exp[q] += rank_contri
                    tmp_exp /= norm_factor
                    possible_exp.append(tmp_exp)

                for k in range(len(candidates)):
                    # for each candidate item
                    if k in selected:
                        continue
                    assume_exp = possible_exp[hqi_sign[k]]
                    disparity = ((np.sqrt(assume_exp) - np.sqrt(self.target_exp)) ** 2).sum() / 2
                    score = lambda_ * tmp_pred[k] - (1 - lambda_) * disparity
                    if score > max_score:
                        max_score = score
                        max_idx = k

                selected.add(max_idx)
                rec_lst[pos] = max_idx
                tmp_pred[max_idx] = -np.inf
                cur_exp[hqi_sign[max_idx]] += rank_contri

            rec_lst[max_topk:] = (-tmp_pred).argsort()[:-max_topk]
            return rec_lst

    torch.multiprocessing.set_sharing_strategy('file_system')
    rerank_dataset = MMRDataset(dataset, predictions)
    dataloader = DataLoader(dataset=rerank_dataset, batch_size=8, num_workers=5)

    sort_idx = list()
    for rerank_res in tqdm(dataloader, leave=False, ncols=100, mininterval=1, desc='Rerank'):
        sort_idx.append(rerank_res.numpy())

    return np.concatenate(sort_idx, axis=0)


# def max_marginal_rel_single(dataset, predictions, max_topk, lambda_=0.5):
#     item2quality = dataset.corpus.item_meta_df['i_quality'].values
#     quality_level = int(item2quality.max() + 1)
#     target_exp = np.ones(quality_level) / quality_level  # Par
#
#     sort_idx = np.zeros_like(predictions, dtype=np.int32)
#     tmp_pred = predictions.copy()
#     for i in tqdm(range(len(dataset)), leave=False, ncols=100, mininterval=1, desc='Rerank'):
#         # for each user
#         selected = set()
#         candidates = dataset[i]['item_id']
#         hqi_sign = [int(item2quality[item - 1]) for item in candidates]
#         cur_exp = np.zeros(quality_level, dtype=np.float)
#         norm_factor = 0
#         for j in range(max_topk):
#             # for each position
#             max_score, max_idx = -np.inf, -1
#             rank_contri = 1 / np.log2(j + 2)
#             norm_factor += rank_contri
#             possible_exp = list()
#             for q in range(quality_level):
#                 tmp_exp = cur_exp.copy()
#                 tmp_exp[q] += rank_contri
#                 tmp_exp /= norm_factor
#                 possible_exp.append(tmp_exp)
#
#             for idx in range(len(candidates))[::-1]:
#                 # for each candidate item
#                 if idx in selected:
#                     continue
#                 disparity = ((np.sqrt(possible_exp[hqi_sign[idx]]) - np.sqrt(target_exp)) ** 2).sum() / 2
#                 score = lambda_ * tmp_pred[i][idx] - (1 - lambda_) * disparity
#                 if score > max_score:
#                     max_score = score
#                     max_idx = idx
#
#             selected.add(max_idx)
#             sort_idx[i][j] = max_idx
#             tmp_pred[i][max_idx] = -np.inf
#             cur_exp[hqi_sign[max_idx]] += rank_contri
#
#     sort_idx[:, max_topk:] = (-tmp_pred).argsort(axis=1)[:, :-max_topk]
#     return sort_idx