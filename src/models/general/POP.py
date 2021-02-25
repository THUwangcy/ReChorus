# -*- coding: UTF-8 -*-

import torch
import numpy as np

from models.BaseModel import GeneralModel


class POP(GeneralModel):
    """
    Recommendation according to item's popularity.
    Should run with --train 0
    """
    def __init__(self, args, corpus):
        self.popularity = np.zeros(corpus.n_items)
        for i in corpus.data_df['train']['item_id'].values:
            self.popularity[i] += 1
        super().__init__(args, corpus)

    def forward(self, feed_dict):
        self.check_list = []
        i_ids = feed_dict['item_id']  # [batch_size, -1]
        prediction = self.popularity[i_ids.cpu().data.numpy()]
        prediction = torch.from_numpy(prediction).to(self.device)
        return {'prediction': prediction.view(feed_dict['batch_size'], -1)}
