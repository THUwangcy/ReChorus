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
from typing import Dict, List, NoReturn

from utils import utils
from models.BaseModel import BaseModel
from helpers.BaseRunner import BaseRunner


class TiMiRunner(BaseRunner):
    def fit(self, data: BaseModel.Dataset, epoch=-1) -> float:
        model = data.model
        if model.optimizer is None:
            model.optimizer = self._build_optimizer(model)
        data.actions_before_epoch()  # must sample before multi thread start

        model.train()
        loss_lst = list()
        dl = DataLoader(data, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers,
                        collate_fn=data.collate_batch, pin_memory=self.pin_memory)
        for batch in tqdm(dl, leave=False, desc='Epoch {:<3}'.format(epoch), ncols=100, mininterval=1):
            batch['epoch'] = epoch
            batch = utils.batch_to_gpu(batch, model.device)
            model.optimizer.zero_grad()
            out_dict = model(batch)
            loss = model.loss(out_dict)
            loss.backward()
            model.optimizer.step()
            loss_lst.append(loss.detach().cpu().data.numpy())
        return np.mean(loss_lst).item()

    def train(self, data_dict: Dict[str, BaseModel.Dataset]) -> NoReturn:
        model = data_dict['train'].model
        main_metric_results, dev_results = list(), list()
        self._check_time(start=True)
        try:
            for epoch in range(self.epoch):
                # Fit
                self._check_time()
                gc.collect()
                torch.cuda.empty_cache()
                loss = self.fit(data_dict['train'], epoch=epoch + 1)
                training_time = self._check_time()

                # Observe selected tensors
                if len(model.check_list) > 0 and self.check_epoch > 0 and epoch % self.check_epoch == 0:
                    utils.check(model.check_list)

                # Record dev results
                dev_result = self.evaluate(data_dict['dev'], self.topk[:1], self.metrics)
                dev_results.append(dev_result)
                main_metric_results.append(dev_result[self.main_metric])
                logging_str = 'Epoch {:<5} loss={:<.4f} [{:<3.1f} s]    dev=({})'.format(
                    epoch + 1, loss, training_time, utils.format_metric(dev_result))

                # Test
                if self.test_epoch > 0 and epoch % self.test_epoch  == 0:
                    test_result = self.evaluate(data_dict['test'], self.topk[:1], self.metrics)
                    logging_str += ' test=({})'.format(utils.format_metric(test_result))
                testing_time = self._check_time()
                logging_str += ' [{:<.1f} s]'.format(testing_time)

                # Save model and early stop
                if max(main_metric_results) == main_metric_results[-1]:
                    model.save_model()
                    logging_str += ' *'
                logging.info(logging_str)

                if self.early_stop > 0 and self.eval_termination(main_metric_results):
                    logging.info("Early stop at %d based on dev result." % (epoch + 1))
                    break
        except KeyboardInterrupt:
            logging.info("Early stop manually")
            exit_here = input("Exit completely without evaluation? (y/n) (default n):")
            if exit_here.lower().startswith('y'):
                logging.info(os.linesep + '-' * 45 + ' END: ' + utils.get_time() + ' ' + '-' * 45)
                exit(1)

        # Find the best dev result across iterations
        best_epoch = main_metric_results.index(max(main_metric_results))
        logging.info(os.linesep + "Best Iter(dev)={:>5}\t dev=({}) [{:<.1f} s] ".format(
            best_epoch + 1, utils.format_metric(dev_results[best_epoch]), self.time[1] - self.time[0]))
        model.load_model()
