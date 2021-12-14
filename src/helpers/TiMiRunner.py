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

    def predict(self, data: BaseModel.Dataset):
        """
        The returned prediction is a 2D-array, each row corresponds to all the candidates,
        and the ground-truth item poses the first.
        Example: ground-truth items: [1, 2], 2 negative items for each instance: [[3,4], [5,6]]
                 predictions like: [[1,3,4], [2,5,6]]
        """
        data.model.eval()
        predictions = list()
        js_div = list()
        dis = list()
        dl = DataLoader(data, batch_size=self.eval_batch_size, shuffle=False, num_workers=self.num_workers,
                        collate_fn=data.collate_batch, pin_memory=self.pin_memory)
        for batch in tqdm(dl, leave=False, ncols=100, mininterval=1, desc='Predict'):
            out_dict = data.model(utils.batch_to_gpu(batch, data.model.device))
            predictions.extend(out_dict['prediction'].cpu().data.numpy())
            if 'js' in out_dict:
                js_div.extend(out_dict['js'].cpu().data.numpy())
            if 'dis' in out_dict:
                dis.extend(out_dict['dis'].cpu().data.numpy())
        if len(js_div) > 0:
            print('JS DIV:', np.mean(js_div))
            np.save('../log/js', np.array(js_div))
        if len(dis) > 0:
            print('Interest Dis:', np.mean(dis))
        return np.array(predictions)
