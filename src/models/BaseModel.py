# -*- coding: UTF-8 -*-

import os
import torch
from tqdm import tqdm
import logging
import numpy as np
import torch.nn.functional as F

from utils import utils


class BaseModel(torch.nn.Module):
    loader = 'BaseLoader'
    runner = 'BaseRunner'
    extra_log_args = []

    @staticmethod
    def parse_model_args(parser, model_name='BaseModel'):
        parser.add_argument('--model_path', type=str, default='',
                            help='Model save path.')
        return parser

    @staticmethod
    def init_weights(m):
        if 'Linear' in str(type(m)):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)
            if m.bias is not None:
                torch.nn.init.normal_(m.bias, mean=0.0, std=0.01)
        elif 'Embedding' in str(type(m)):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)

    def __init__(self, model_path):
        super(BaseModel, self).__init__()
        self.model_path = model_path
        self.batches_buffer = dict()  # save batches of dev and test set
        self.check_list = list()  # observe tensors in check_list every check_epoch
        self.embedding_l2 = list()  # manually calculate l2 of used embeddings in the list, not necessary to maintain

        self._define_params()
        self.total_parameters = self.count_variables()
        logging.info('#params: %d' % self.total_parameters)

        self.optimizer = None

    """
    Methods must to override
    """
    def _define_params(self):
        self.item_bias = torch.nn.Embedding(self.item_num, 1)
        self.embeddings = ['item_bias']  # exclude direct l2 calculation of these embeddings, not necessary to maintain

    def forward(self, feed_dict):
        self.check_list, self.embedding_l2 = [], []
        i_ids = feed_dict['item_id']
        i_bias = self.item_bias(i_ids)
        self.embedding_l2.append(i_bias)
        out_dict = {'prediction': i_bias.view(feed_dict['batch_size'], -1), 'check': self.check_list}
        return out_dict

    def get_feed_dict(self, corpus, data, batch_start, batch_size, phase):
        """
        Generate a batch of the given data, which will be fed into forward function.
        :param corpus: Loader object
        :param data: DataFrame in corpus.data_df (may be shuffled)
        :param batch_start: batch start index
        :param batch_size: batch size
        :param phase: 'train', 'dev' or 'test'
        """
        batch_end = min(len(data), batch_start + batch_size)
        real_batch_size = batch_end - batch_start
        item_ids = data['item_id'][batch_start: batch_start + real_batch_size].values
        neg_items = self.get_neg_items(corpus, data, batch_start, real_batch_size, phase)  # [batch_size, num_neg]
        # concatenate ground-truth item and corresponding negative items
        item_ids = np.concatenate([np.expand_dims(item_ids, -1), neg_items], axis=1)
        feed_dict = {'item_id': utils.numpy_to_torch(item_ids), 'batch_size': real_batch_size}
        return feed_dict

    """
    Methods optional to override
    """
    def prepare_batches(self, corpus, data, batch_size, phase):
        buffer_key = '_'.join([phase, str(batch_size)])
        if buffer_key in self.batches_buffer:
            return self.batches_buffer[buffer_key]

        # generate the list of all batches of the given data
        # TODO: multi-thread preparation
        num_example = len(data)
        total_batch = int((num_example + batch_size - 1) / batch_size)
        batches = list()
        for batch in tqdm(range(total_batch), leave=False, ncols=100, mininterval=1, desc='Prepare Batches'):
            batches.append(self.get_feed_dict(corpus, data, batch * batch_size, batch_size, phase))

        if phase != 'train':
            self.batches_buffer[buffer_key] = batches
        return batches

    def get_neg_items(self, corpus, data, batch_start, real_batch_size, phase):
        if phase == 'train':  # for training, sample paired negative items that haven't been interacted
            user_ids = data['user_id'][batch_start: batch_start + real_batch_size].values
            neg_items = np.random.randint(1, self.item_num, size=(real_batch_size, 1))
            for i in range(real_batch_size):
                while neg_items[i][0] in corpus.user_clicked_set[user_ids[i]]:
                    neg_items[i][0] = np.random.randint(1, self.item_num)
        else:  # for dev and test, negative items are prepared in advance
            neg_items = data['neg_items'][batch_start: batch_start + real_batch_size].tolist()
        return neg_items

    def loss(self, feed_dict, predictions):
        # BPR ranking loss. For numerical stability, we use '-softplus(-x)' instead of 'log_sigmoid(x)'
        pos_pred, neg_pred = predictions[:, 0], predictions[:, 1]
        loss = F.softplus(-(pos_pred - neg_pred)).mean()
        return loss

    def customize_parameters(self):
        # customize optimizer settings for different parameters
        weight_p, bias_p = [], []
        for name, p in filter(lambda x: x[1].requires_grad, self.named_parameters()):
            if 'bias' in name:
                bias_p.append(p)
            else:
                weight_p.append(p)
        optimize_dict = [{'params': weight_p}, {'params': bias_p, 'weight_decay': 0.0}]
        return optimize_dict

    def actions_before_train(self):
        pass

    def actions_after_train(self):
        pass

    """"""

    def count_variables(self):
        total_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_parameters

    def l2(self):
        # calculate l2 of a batch manually for observation
        l2 = utils.numpy_to_torch(np.array(0.0, dtype=np.float64), gpu=True)
        for name, p in filter(lambda x: x[1].requires_grad, self.named_parameters()):
            if ('bias' not in name) and (name.split('.')[0] not in self.embeddings):
                l2 += (p ** 2).sum()
        # only include embeddings utilized in the current batch
        for p in self.embedding_l2:
            l2 += (p ** 2).sum() / p.shape[0]
        return l2

    def check(self, out_dict):
        # observe selected tensors during forward.
        logging.info('')
        for i, t in enumerate(self.check_list):
            d = np.array(t[1].detach().cpu())
            logging.info(os.linesep.join(
                [t[0] + '\t' + str(d.shape), np.array2string(d, threshold=20)]
            ) + os.linesep)
        loss, l2 = out_dict['mean_loss'], out_dict['mean_l2']
        logging.info('loss = %.4f, l2 = %.4f' % (loss, l2))

    def save_model(self, model_path=None):
        if model_path is None:
            model_path = self.model_path
        utils.check_dir(model_path)
        torch.save(self.state_dict(), model_path)
        logging.info('Save model to ' + model_path[:50] + '...')

    def load_model(self, model_path=None):
        if model_path is None:
            model_path = self.model_path
        self.load_state_dict(torch.load(model_path))
        logging.info('Load model from ' + model_path)
