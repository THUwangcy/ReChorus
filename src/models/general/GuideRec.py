# -*- coding: UTF-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import logging
import numpy as np

from models.BaseModel import GeneralModel


class GuideRec(GeneralModel):
    reader = 'GuideReader'
    runner = 'GuideRunner'
    extra_log_args = ['emb_size', 'gamma', 'sample_hqi', 'temperature', 'stage']

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--emb_size', type=int, default=64,
                            help='Size of embedding vectors.')
        parser.add_argument('--gamma', type=float, default=0.1,
                            help='Weight of the hqi to be mixed.')
        parser.add_argument('--sample_hqi', type=str, default='random',
                            help='Strategy to sample hqi: random, pos_prob')
        parser.add_argument('--temperature', type=float, default=1,
                            help='Sample weight temperature for softmax')
        parser.add_argument('--stage', type=str, default='finetune',
                            help='Training stage: pretrain / finetune.')

        return GeneralModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        super().__init__(args, corpus)
        self.emb_size = args.emb_size
        self.gamma = args.gamma
        self.sample_hqi = args.sample_hqi
        self.temperature = args.temperature
        self.stage = args.stage
        self._define_params()
        self.apply(self.init_weights)

        self.backbone_path = '../model/GuideRec/BPRMF__{}__{}__emb_size={}.pt' \
            .format(corpus.dataset, args.random_seed, self.emb_size)
        if self.stage == 'pretrain':
            self.model_path = self.backbone_path
        elif self.stage == 'finetune' and os.path.exists(self.backbone_path):
            self.load_model(self.backbone_path)
        else:
            logging.info('Train from scratch!')

    def _define_params(self):
        self.u_embeddings = nn.Embedding(self.user_num, self.emb_size)
        self.i_embeddings = nn.Embedding(self.item_num, self.emb_size)

    def load_model(self, model_path=None):
        if model_path is None:
            model_path = self.model_path
        model_dict = self.state_dict()
        state_dict = torch.load(model_path)
        exist_state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
        model_dict.update(exist_state_dict)
        self.load_state_dict(model_dict)
        logging.info('Load model from ' + model_path)

    @staticmethod
    def distance(x, y):
        x, y = F.normalize(x, dim=-1), F.normalize(y, dim=-1)
        return (x - y).norm(p=2, dim=-1)

    @staticmethod
    def similarity(x, y):
        x, y = F.normalize(x, dim=-1), F.normalize(y, dim=-1)
        return (x * y).sum(dim=-1)

    def forward(self, feed_dict):
        self.check_list = []
        u_ids = feed_dict['user_id']  # [batch_size]
        i_ids = feed_dict['item_id']  # [batch_size, -1]

        user_e = self.u_embeddings(u_ids)
        item_e = self.i_embeddings(i_ids)

        # mix: mix pos item and high quality item
        if self.stage != 'pretrain' and feed_dict['phase'] == 'train':
            hqi_ids = feed_dict['hqi_id']  # [batch_size]
            hqi_e = self.i_embeddings(hqi_ids.long())
            item_e[:, 0, :] = (1 - self.gamma) * item_e[:, 0, :] + self.gamma * hqi_e

        prediction = (user_e[:, None, :] * item_e).sum(dim=-1)  # [batch_size, -1]
        return {'prediction': prediction.view(feed_dict['batch_size'], -1)}

    class Dataset(GeneralModel.Dataset):
        def _get_feed_dict(self, index):
            feed_dict = super()._get_feed_dict(index)
            if self.model.stage != 'pretrain' and self.phase == 'train':  # add hqi_id to feed_dict
                hqi_id = self.data['hqis'][index]
                feed_dict.update({
                    'hqi_id': hqi_id
                })
            return feed_dict

        # Negtive items and High Quality item
        @torch.no_grad()
        def actions_before_epoch(self):
            super().actions_before_epoch()

            if self.model.stage == 'pretrain':
                return

            if self.model.sample_hqi == 'random':
                self.data['hqis'] = np.array(np.random.choice(list(self.corpus.HQI_set), size=len(self)))

            elif self.model.sample_hqi == 'pos_prob':
                hqi_ids = torch.tensor(list(self.corpus.HQI_set))  # [len(hqi_set)]
                hqis_e = self.model.i_embeddings(hqi_ids.to(self.model.device))
                items_e = self.model.i_embeddings.weight
                norm_hqis_e = F.normalize(hqis_e, dim=-1)
                norm_items_e = F.normalize(items_e, dim=-1)
                i_sim = torch.matmul(norm_items_e, norm_hqis_e.T) / self.model.temperature
                i_sim = i_sim.softmax(dim=-1).cpu()
                self.data['hqis'] = np.zeros(len(self), dtype=np.int)
                for idx, (i, u) in enumerate(
                        zip(self.data['item_id'], self.data['user_id'])):  # sample hqi for each pos_id
                    i_weight = i_sim[i].clone()
                    clicked_set = self.corpus.train_clicked_set[u]
                    sample_idx = torch.multinomial(i_weight, num_samples=1, replacement=False).view(-1)
                    sampled_hqi = hqi_ids[sample_idx]
                    while sampled_hqi in clicked_set:  # filter hqi in user's clicked_set
                        i_weight[sample_idx] = 0
                        sample_idx = torch.multinomial(i_weight, num_samples=1, replacement=False).view(-1)
                        sampled_hqi = hqi_ids[sample_idx]
                    self.data['hqis'][idx] = sampled_hqi

                # hqi_ids = torch.tensor(list(self.corpus.HQI_set)).to(self.model.device)  # [len(hqi_set)]
                # hqis_e = self.model.i_embeddings(hqi_ids).data
                # items_e = self.model.i_embeddings.weight.data
                # self.data['hqis'] = np.zeros(len(self), dtype=np.int)
                # hqi_ids = hqi_ids.cpu()
                # for idx, (i, u) in enumerate(
                #         zip(self.data['item_id'], self.data['user_id'])):  # sample hqi for each pos_id
                #     i_distance = GuideRec.distance(items_e[i], hqis_e)
                #     clicked_set = self.corpus.train_clicked_set[u]
                #
                #     min_idx = torch.multinomial(i_distance.max() - i_distance, num_samples=1, replacement=True).view(-1)
                #     sampled_hqi = hqi_ids[min_idx].data
                #     while sampled_hqi in clicked_set:  # filter hqi in user's clicked_set
                #         i_distance[min_idx] = i_distance.max()
                #         min_idx = torch.multinomial(i_distance.max() - i_distance, num_samples=1,
                #                                     replacement=True).view(-1)
                #         sampled_hqi = hqi_ids[min_idx].data
                #     self.data['hqis'][idx] = sampled_hqi

            else:
                raise ValueError('Undefined sample_hqi strategy: {}.'.format(self.model.sample_hqi))

            # check sampled hqis
            # sampled_hqi_set = set(np.array(self.data['hqis']))
            # print("#hqi:{}, #hqi_pop:{}".format(len(sampled_hqi_set), len(self.corpus.P_set & sampled_hqi_set)))
