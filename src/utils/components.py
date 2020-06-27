# -*- coding: UTF-8 -*-

import torch
import numpy as np


def scaled_dot_product_attention(q, k, v, scale=None, attn_mask=None):
    """
    Weighted sum of v according to dot product between q and k
    :param q: query tensor，[-1, L_q, V]
    :param k: key tensor，[-1, L_k, V]
    :param v: value tensor，[-1, L_k, V]
    :param scale: scalar
    :param attn_mask: [-1, L_q, L_k]
    """
    attention = torch.bmm(q, k.transpose(1, 2))  # [-1, L_q, L_k]
    if scale is not None:
        attention = attention * scale
    attention = attention - attention.max()
    if attn_mask is not None:
        attention = attention.masked_fill(attn_mask, -np.inf)
    attention = attention.softmax(dim=-1)
    context = torch.bmm(attention, v)  # [-1, L_q, V]
    return context
