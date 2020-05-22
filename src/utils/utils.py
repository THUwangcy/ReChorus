# -*- coding: UTF-8 -*-

import os
import datetime
import torch
import numpy as np


def get_time():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_hr(gt_rank, topk):
    """
    Only for the situation when there is one ground-truth item.
    :param gt_rank: the position of ground-truth item in ranking list
    :return: HR@topk
    """
    return int(gt_rank <= topk)


def get_ndcg(gt_rank, topk):
    """
    Only for the situation when there is one ground-truth item.
    :param gt_rank: the position of ground-truth item in ranking list
    :return: NDCG@topk
    """
    return int(gt_rank <= topk) / np.log2(gt_rank + 1)


def topk_evaluate_method(predictions, topk, metrics):
    """
    :param predictions: (-1, n_candidates) shape, the first column is the score for ground-truth item
    :param topk: top-K list
    :param metrics: metrics dict
    :return: a result dict, the keys are metrics
    """
    evaluations = dict()
    for k in topk:
        for metric in metrics:
            evaluations['{}@{}'.format(metric, k)] = list()
    for prediction in predictions:
        gt_rank = np.argsort(prediction)[::-1].tolist().index(0) + 1
        for k in topk:
            for metric in metrics:
                m = '{}@{}'.format(metric, k)
                if metric == 'HR':
                    evaluations[m].append(get_hr(gt_rank, k))
                elif metric == 'NDCG':
                    evaluations[m].append(get_ndcg(gt_rank, k))
                else:
                    raise ValueError('Undefined evaluation metric: {}.'.format(metric))
    for k in topk:
        for metric in metrics:
            m = '{}@{}'.format(metric, k)
            evaluations[m] = np.mean(evaluations[m])
    return evaluations


def numpy_to_torch(d, gpu=True):
    t = torch.from_numpy(d)
    if gpu and torch.cuda.device_count() > 0:
        t = t.cuda()
    return t


def pad_lst(lst, dtype=np.int64):
    inner_max_len = max(map(len, lst))
    result = np.zeros([len(lst), inner_max_len], dtype)
    for i, row in enumerate(lst):
        for j, val in enumerate(row):
            result[i][j] = val
    return result


def format_metric(metric):
    assert type(metric) == dict
    format_str = []
    for name in np.sort(list(metric.keys())):
        m = metric[name]
        if type(m) is float or type(m) is np.float or type(m) is np.float32 or type(m) is np.float64:
            format_str.append('{}:{:<.4f}'.format(name, m))
        elif type(m) is int or type(m) is np.int or type(m) is np.int32 or type(m) is np.int64:
            format_str.append('{}:{}'.format(name, m))
    return ','.join(format_str)


def format_arg_str(args, max_len=20):
    linesep = os.linesep
    arg_dict = vars(args)
    keys, values = arg_dict.keys(), arg_dict.values()
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


def non_increasing(l):
    return all(x >= y for x, y in zip(l, l[1:]))
