# -*- coding: UTF-8 -*-

import os
import logging
import torch
import datetime
import numpy as np


def evaluate_method(predictions, topk, metrics):
    """
    :param predictions: (-1, n_candidates) shape, the first column is the score for ground-truth item
    :param topk: top-K values list
    :param metrics: metrics string list
    :return: a result dict, the keys are metrics@topk
    """
    evaluations = dict()
    sort_idx = (-predictions).argsort(axis=1)
    gt_rank = np.argwhere(sort_idx == 0)[:, 1] + 1
    for k in topk:
        hit = (gt_rank <= k)
        for metric in metrics:
            key = '{}@{}'.format(metric, k)
            if metric == 'HR':
                evaluations[key] = hit.mean()
            elif metric == 'NDCG':
                evaluations[key] = (hit / np.log2(gt_rank + 1)).mean()
            else:
                raise ValueError('Undefined evaluation metric: {}.'.format(metric))
    return evaluations


def df_to_dict(df):
    res = df.to_dict('list')
    for key in res:
        res[key] = np.array(res[key])
    return res


def numpy_to_torch(d, gpu=True):
    t = torch.from_numpy(d)
    if gpu and torch.cuda.device_count() > 0:
        t = t.cuda()
    return t


def batch_to_gpu(batch):
    if torch.cuda.device_count() > 0:
        for c in batch:
            if type(batch[c]) is torch.Tensor:
                batch[c] = batch[c].cuda()
    return batch


def check(check_list):
    # observe selected tensors during training.
    logging.info('')
    for i, t in enumerate(check_list):
        d = np.array(t[1].detach().cpu())
        logging.info(os.linesep.join(
            [t[0] + '\t' + str(d.shape), np.array2string(d, threshold=20)]
        ) + os.linesep)


def format_metric(result_dict):
    assert type(result_dict) == dict
    format_str = []
    for name in np.sort(list(result_dict.keys())):
        m = result_dict[name]
        if type(m) is float or type(m) is np.float or type(m) is np.float32 or type(m) is np.float64:
            format_str.append('{}:{:<.4f}'.format(name, m))
        elif type(m) is int or type(m) is np.int or type(m) is np.int32 or type(m) is np.int64:
            format_str.append('{}:{}'.format(name, m))
    return ','.join(format_str)


def format_arg_str(args, exclude_lst, max_len=20):
    linesep = os.linesep
    arg_dict = vars(args)
    keys = [k for k in arg_dict.keys() if k not in exclude_lst]
    values = [arg_dict[k] for k in keys]
    key_title, value_title = 'Arguments', 'Values'
    key_max_len = max(map(lambda x: len(str(x)), keys))
    value_max_len = min(max(map(lambda x: len(str(x)), values)), max_len)
    key_max_len, value_max_len = max([len(key_title), key_max_len]), max([len(value_title), value_max_len])
    horizon_len = key_max_len + value_max_len + 5
    res_str = linesep + '=' * horizon_len + linesep
    res_str += ' ' + key_title + ' ' * (key_max_len - len(key_title)) + ' | ' \
               + value_title + ' ' * (value_max_len - len(value_title)) + ' ' + linesep + '=' * horizon_len + linesep
    for key in sorted(keys):
        value = arg_dict[key]
        if value is not None:
            key, value = str(key), str(value).replace('\t', '\\t')
            value = value[:max_len-3] + '...' if len(value) > max_len else value
            res_str += ' ' + key + ' ' * (key_max_len - len(key)) + ' | ' \
                       + value + ' ' * (value_max_len - len(value)) + linesep
    res_str += '=' * horizon_len
    return res_str


def check_dir(file_name):
    dir_path = os.path.dirname(file_name)
    if not os.path.exists(dir_path):
        print('make dirs:', dir_path)
        os.makedirs(dir_path)


def non_increasing(lst):
    return all(x >= y for x, y in zip(lst, lst[1:]))


def get_time():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
