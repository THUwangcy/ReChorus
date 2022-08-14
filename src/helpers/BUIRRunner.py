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

from utils import utils
from models.BaseModel import BaseModel
from helpers.BaseRunner import BaseRunner


class BUIRRunner(BaseRunner):
    def fit(self, dataset: BaseModel.Dataset, epoch=-1) -> float:
        model = dataset.model
        if model.optimizer is None:
            model.optimizer = self._build_optimizer(model)
        dataset.actions_before_epoch()  # must sample before multi thread start

        model.train()
        loss_lst = list()
        dl = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers,
                        collate_fn=dataset.collate_batch, pin_memory=self.pin_memory)
        for batch in tqdm(dl, leave=False, desc='Epoch {:<3}'.format(epoch), ncols=100, mininterval=1):
            batch = utils.batch_to_gpu(batch, model.device)
            model.optimizer.zero_grad()
            out_dict = model(batch)
            loss = model.loss(out_dict)
            loss.backward()
            model.optimizer.step()
            model._update_target()
            loss_lst.append(loss.detach().cpu().data.numpy())
        return np.mean(loss_lst).item()
