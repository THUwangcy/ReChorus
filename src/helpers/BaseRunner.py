# -*- coding: UTF-8 -*-

import torch
import logging
from time import time
from tqdm import tqdm
import gc
import numpy as np
import copy
import os

from utils import utils


class BaseRunner(object):
    @staticmethod
    def parse_runner_args(parser):
        parser.add_argument('--epoch', type=int, default=100,
                            help='Number of epochs.')
        parser.add_argument('--check_epoch', type=int, default=10,
                            help='Check every epochs.')
        parser.add_argument('--early_stop', type=int, default=1,
                            help='whether to early-stop.')
        parser.add_argument('--lr', type=float, default=1e-3,
                            help='Learning rate.')
        parser.add_argument('--batch_size', type=int, default=256,
                            help='Batch size during training.')
        parser.add_argument('--eval_batch_size', type=int, default=256,
                            help='Batch size during testing.')
        parser.add_argument('--dropout', type=float, default=0.2,
                            help='Dropout probability for each deep layer')
        parser.add_argument('--l2', type=float, default=0,
                            help='Weight decay in optimizer.')
        parser.add_argument('--optimizer', type=str, default='Adam',
                            help='optimizer: GD, Adam, Adagrad, Adadelta')
        parser.add_argument('--topk', type=str, default='[5,10]',
                            help='The number of items recommended to each user.')
        parser.add_argument('--metric', type=str, default='["NDCG","HR"]',
                            help='metrics: NDCG, HR')
        return parser

    def __init__(self, args):
        self.epoch = args.epoch
        self.check_epoch = args.check_epoch
        self.early_stop = args.early_stop
        self.learning_rate = args.lr
        self.batch_size = args.batch_size
        self.eval_batch_size = args.eval_batch_size
        self.l2 = args.l2
        self.optimizer_name = args.optimizer
        self.topk = eval(args.topk)
        self.metrics = [m.strip().upper() for m in eval(args.metric)]

        self.time = None  # will store [start_time, last_step_time]
        self.main_metric = self.metrics[0]
        self.main_topk = self.topk[0]
        self.main_key = '{}@{}'.format(self.main_metric, self.main_topk)  # early stop based on main_key
        self.dev_results, self.test_results = list(), list()

    def _build_optimizer(self, model):
        optimizer_name = self.optimizer_name.lower()
        if optimizer_name == 'gd':
            logging.info("Optimizer: GD")
            optimizer = torch.optim.SGD(
                model.customize_parameters(), lr=self.learning_rate, weight_decay=self.l2
            )
        elif optimizer_name == 'adagrad':
            logging.info("Optimizer: Adagrad")
            optimizer = torch.optim.Adagrad(
                model.customize_parameters(), lr=self.learning_rate, weight_decay=self.l2
            )
        elif optimizer_name == 'adadelta':
            logging.info("Optimizer: Adadelta")
            optimizer = torch.optim.Adadelta(
                model.customize_parameters(), lr=self.learning_rate, weight_decay=self.l2
            )
        elif optimizer_name == 'adam':
            logging.info("Optimizer: Adam")
            optimizer = torch.optim.Adam(
                model.customize_parameters(), lr=self.learning_rate, weight_decay=self.l2
            )
        else:
            raise ValueError("Unknown Optimizer: " + self.optimizer_name)
        return optimizer

    def _check_time(self, start=False):
        if self.time is None or start:
            self.time = [time()] * 2
            return self.time[0]
        tmp_time = self.time[1]
        self.time[1] = time()
        return self.time[1] - tmp_time

    def train(self, model, corpus):
        assert corpus.data_df['train'] is not None
        self._check_time(start=True)

        try:
            for epoch in range(self.epoch):
                self._check_time()
                # Shuffle training data
                epoch_train_data = copy.deepcopy(corpus.data_df['train'])
                epoch_train_data = epoch_train_data.sample(frac=1).reset_index(drop=True)

                # Fit
                last_batch, mean_loss, mean_l2 = self.fit(model, corpus, epoch_train_data, epoch=epoch + 1)

                # Observe selective tensors
                if self.check_epoch > 0 and epoch % self.check_epoch == 0:
                    last_batch['mean_loss'] = mean_loss
                    last_batch['mean_l2'] = mean_l2
                    model.check(last_batch)
                del epoch_train_data
                training_time = self._check_time()

                # Record dev and test results
                dev_result = self.evaluate(model, corpus, 'dev', [self.main_topk], self.metrics)
                test_result = self.evaluate(model, corpus, 'test', [self.main_topk], self.metrics)
                testing_time = self._check_time()
                self.dev_results.append(dev_result)
                self.test_results.append(test_result)

                logging.info("Epoch {:<3} loss={:<.4f} [{:<.1f} s]\t dev=({}) test=({}) [{:<.1f} s] ".format(
                             epoch + 1, mean_loss, training_time, utils.format_metric(dev_result),
                             utils.format_metric(test_result), testing_time))

                # Save model and early stop
                main_metric_result = [x[self.main_key] for x in self.dev_results]
                if max(main_metric_result) == main_metric_result[-1] \
                        or (hasattr(model, 'stage') and model.stage == 1):
                    model.save_model()
                if self.early_stop and self.eval_termination():
                    logging.info("Early stop at %d based on validation result." % (epoch + 1))
                    break
        except KeyboardInterrupt:
            logging.info("Early stop manually")
            exit_here = input("Exit completely without evaluation? (y/n) (default n):")
            if exit_here.lower().startswith('y'):
                logging.info(os.linesep + '-' * 45 + ' END: ' + utils.get_time() + ' ' + '-' * 45)
                exit(1)

        # Find the best dev result across iterations
        main_metric_result = [x[self.main_key] for x in self.dev_results]
        best_dev_score = max(main_metric_result)
        best_epoch = main_metric_result.index(best_dev_score)
        logging.info("\nBest Iter(dev)=  %5d\t dev=(%s) test=(%s) [%.1f s] "
                     % (best_epoch + 1, utils.format_metric(self.dev_results[best_epoch]),
                        utils.format_metric(self.test_results[best_epoch]), self.time[1] - self.time[0]))
        model.load_model()

    def fit(self, model, corpus, epoch_train_data, epoch=-1):
        gc.collect()
        torch.cuda.empty_cache()
        if model.optimizer is None:
            model.optimizer = self._build_optimizer(model)
        batches = model.prepare_batches(corpus, epoch_train_data, self.batch_size, phase='train')

        model.train()
        loss_lst, l2_lst, output_dict = list(), list(), None
        for batch in tqdm(batches, leave=False, desc='Epoch {:<3}'.format(epoch), ncols=100, mininterval=1):
            model.optimizer.zero_grad()
            output_dict = model(batch)
            loss = model.loss(batch, output_dict['prediction'])
            l2 = model.l2() * self.l2
            loss.backward()
            model.optimizer.step()
            loss_lst.append(loss.detach().cpu().data.numpy())
            l2_lst.append(l2.detach().cpu().data.numpy())
        return output_dict, np.mean(loss_lst), np.mean(l2_lst)

    def eval_termination(self):
        dev = [x[self.main_key] for x in self.dev_results]
        if len(dev) > 20 and utils.non_increasing(dev[-5:]):
            return True
        elif len(dev) - dev.index(max(dev)) > 20:
            return True
        return False

    def evaluate(self, model, corpus, phase, topks, metrics):
        """
        Evaluate the results for an input set.
        :return: result dict (key: metric@k)
        """
        predictions = self.predict(model, corpus, phase)
        return utils.topk_evaluate_method(predictions, topks, metrics)

    def predict(self, model, corpus, phase):
        """
        The returned prediction is a 2D-array, each row corresponds to all the candidates,
        and the ground-truth item poses the first.
        Example: ground-truth items: [1, 2], 2 negative items for each instance: [[3,4], [5,6]]
                 predictions order: [[1,3,4], [2,5,6]]
        """
        gc.collect()
        torch.cuda.empty_cache()
        batches = model.prepare_batches(corpus, corpus.data_df[phase], self.eval_batch_size, phase=phase)

        model.eval()
        predictions = list()
        for batch in tqdm(batches, leave=False, ncols=100, mininterval=1, desc='Predict'):
            prediction = model(batch)['prediction']
            predictions.extend(prediction.cpu().data.numpy())
        return np.array(predictions)

    def print_res(self, model, corpus):
        """
        Construct the final test result string before/after training
        :return: test result string
        """
        phase, res_str = 'test', ''
        result_dict = self.evaluate(model, corpus, phase, self.topk, self.metrics)
        res_str += '(' + utils.format_metric(result_dict) + ')'
        return res_str
