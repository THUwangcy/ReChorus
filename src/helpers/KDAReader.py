# -*- coding: UTF-8 -*-

import os
import pickle
import argparse
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm

from helpers.KGReader import KGReader


""" Data Reading for KDA """
class KDAReader(KGReader):
    @staticmethod
    def parse_data_args(parser):
        parser.add_argument('--t_scalar', type=int, default=60,
                            help='Time interval scalar.')
        parser.add_argument('--n_dft', type=int, default=64,
                            help='The point of DFT.')
        parser.add_argument('--freq_rand', type=int, default=0,
                            help='Whether randomly initialize parameters in frequency domain.')
        return KGReader.parse_data_args(parser)

    @staticmethod
    def dft(x: list, n_dft=-1) -> np.ndarray:
        if n_dft <= 0:
            n_dft = 2 ** (int(np.log2(len(x))) + 1)
        freq_x = np.fft.fft(x, n_dft)
        return 2 * freq_x[: n_dft // 2 + 1]  # fold negative frequencies

    @staticmethod
    def norm_time(a: list, t_scalar: int) -> np.ndarray:
        norm_t = np.log2(np.array(a) / t_scalar + 1e-6)
        norm_t = np.maximum(norm_t, 0)
        return norm_t

    def __init__(self, args):
        super().__init__(args)
        self.t_scalar = args.t_scalar
        self.n_dft = args.n_dft
        self.freq_rand = args.freq_rand
        self.regenerate = args.regenerate
        self.interval_file = os.path.join(self.prefix, self.dataset, 'interval.pkl')

        self.freq_x = np.empty((self.n_relations, self.n_dft // 2 + 1), dtype=complex)
        if not self.freq_rand:
            self._time_interval_cnt()  # ! May need a lot of time
            self._cal_freq_x()

    # Calculate time intervals of relational neighbors for each relation type (include a virtual relation)
    def _time_interval_cnt(self):
        if os.path.exists(self.interval_file) and not self.regenerate:
            self.interval_dict = pickle.load(open(self.interval_file, 'rb'))
            return

        self.interval_dict = {'virtual': list()}
        for relation_type in self.relations:
            self.interval_dict[relation_type] = list()

        merge_df = pd.merge(self.all_df, self.item_meta_df, how='left', on='item_id')
        gb_user = merge_df.groupby('user_id')
        for user, user_df in tqdm(gb_user, leave=False, ncols=100, mininterval=1, desc='Count Intervals'):
            # Virtual item-item relation
            times, iids = user_df['time'].values, user_df['item_id'].values
            delta_t = [t for t in (times[1:] - times[:-1]) if t > 0]
            self.interval_dict['virtual'].extend(delta_t)
            # Attribute based relations
            for attr in self.attr_relations:
                for val, df in user_df.groupby(attr):
                    delta_t = [t for t in (df['time'].values[1:] - df['time'].values[:-1]) if t > 0]
                    self.interval_dict[attr].extend(delta_t)
            # Natural item relations
            for r_idx, relation in enumerate(self.item_relations):
                for target_idx in range(1, len(iids))[::-1]:  # traverse tail item back-to-front in user history
                    target_i, target_t = iids[target_idx], times[target_idx]
                    for source_idx in range(target_idx)[::-1]:  # look forward from the tail item
                        source_i, source_t = iids[source_idx], times[source_idx]
                        delta_t = target_t - source_t
                        if delta_t > 0 and (source_i, r_idx + 1, target_i) in self.triplet_set:
                            self.interval_dict[relation].append(delta_t)
                            break

        pickle.dump(self.interval_dict, open(self.interval_file, 'wb'))

    # Apply DFT on time interval distributions to get initial frequency representations
    def _cal_freq_x(self):
        distributions = list()
        # normalized time interval distributions
        for col in ['virtual'] + self.relations:
            intervals = self.norm_time(self.interval_dict[col], self.t_scalar)
            bin_num = int(max(intervals)) + 1
            ns = np.zeros(bin_num)
            for inter in intervals:
                ns[int(inter)] += 1
            distributions.append(ns / max(ns))
            min_dft = 2 ** (int(np.log2(bin_num) + 1))
            if self.n_dft < min_dft:
                self.n_dft = min_dft
        # DFT
        for i, dist in enumerate(distributions):
            dft_res = self.dft(dist, self.n_dft)
            self.freq_x[i] = dft_res

        del self.interval_dict
