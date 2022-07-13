# -*- coding: UTF-8 -*-

import os
import sys
import pickle
import logging
import argparse
import pandas as pd
import torch

from helpers import *
from models.general import *
from models.sequential import *
from models.developing import *
from utils import utils


def parse_global_args(parser):
    parser.add_argument('--gpu', type=str, default='',
                        help='Set CUDA_VISIBLE_DEVICES, default for CPU only')
    parser.add_argument('--verbose', type=int, default=logging.INFO,
                        help='Logging Level, 0, 10, ..., 50')
    parser.add_argument('--log_file', type=str, default='',
                        help='Logging file path')
    parser.add_argument('--random_seed', type=int, default=0,
                        help='Random seed of numpy and pytorch')
    parser.add_argument('--load', type=int, default=0,
                        help='Whether load model and continue to train')
    parser.add_argument('--train', type=int, default=1,
                        help='To train the model or not.')
    parser.add_argument('--regenerate', type=int, default=0,
                        help='Whether to regenerate intermediate files')
    return parser


def main():
    logging.info('-' * 45 + ' BEGIN: ' + utils.get_time() + ' ' + '-' * 45)
    exclude = ['check_epoch', 'log_file', 'model_path', 'path', 'pin_memory', 'load',
               'regenerate', 'sep', 'train', 'verbose', 'metric', 'test_epoch', 'buffer']
    logging.info(utils.format_arg_str(args, exclude_lst=exclude))

    # Random seed
    utils.init_seed(args.random_seed)

    # GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.device = torch.device('cpu')
    if args.gpu != '' and torch.cuda.is_available():
        args.device = torch.device('cuda')
    logging.info('Device: {}'.format(args.device))

    # Read data
    corpus_path = os.path.join(args.path, args.dataset, model_name.reader + '.pkl')
    if not args.regenerate and os.path.exists(corpus_path):
        logging.info('Load corpus from {}'.format(corpus_path))
        corpus = pickle.load(open(corpus_path, 'rb'))
    else:
        corpus = reader_name(args)
        logging.info('Save corpus to {}'.format(corpus_path))
        pickle.dump(corpus, open(corpus_path, 'wb'))

    # Define model
    model = model_name(args, corpus).to(args.device)
    logging.info('#params: {}'.format(model.count_variables()))
    logging.info(model)

    # Run model
    data_dict = dict()
    for phase in ['train', 'dev', 'test']:
        data_dict[phase] = model_name.Dataset(model, corpus, phase)
        data_dict[phase].prepare()
    runner = runner_name(args)
    # logging.info('Test Before Training: ' + runner.print_res(data_dict['test']))
    if args.load > 0:
        model.load_model()
    if args.train > 0:
        runner.train(data_dict)
    eval_res = runner.print_res(data_dict['test'])
    logging.info(os.linesep + 'Test After Training: ' + eval_res)
    # save_rec_results(data_dict['dev'], runner, 100)
    model.actions_after_train()
    logging.info(os.linesep + '-' * 45 + ' END: ' + utils.get_time() + ' ' + '-' * 45)


def save_rec_results(dataset, runner, topk):
    result_path = os.path.join(args.path, args.dataset, 'rec-{}.csv'.format(init_args.model_name))
    logging.info('Saving top-{} recommendation results to: {}'.format(topk, result_path))
    predictions = runner.predict(dataset)  # n_users, n_candidates
    users, rec_items = list(), list()
    for i in range(len(dataset)):
        info = dataset[i]
        users.append(info['user_id'])
        item_scores = zip(info['item_id'], predictions[i])
        sorted_lst = sorted(item_scores, key=lambda x: x[1], reverse=True)[:topk]
        rec_items.append([x[0] for x in sorted_lst])
    rec_df = pd.DataFrame(columns=['user_id', 'rec_items'])
    rec_df['user_id'] = users
    rec_df['rec_items'] = rec_items
    rec_df.to_csv(result_path, sep=args.sep, index=False)


if __name__ == '__main__':
    init_parser = argparse.ArgumentParser(description='Model')
    init_parser.add_argument('--model_name', type=str, default='BPRMF', help='Choose a model to run.')
    init_args, init_extras = init_parser.parse_known_args()
    model_name = eval('{0}.{0}'.format(init_args.model_name))
    reader_name = eval('{0}.{0}'.format(model_name.reader))  # model chooses the reader
    runner_name = eval('{0}.{0}'.format(model_name.runner))  # model chooses the runner

    # Args
    parser = argparse.ArgumentParser(description='')
    parser = parse_global_args(parser)
    parser = reader_name.parse_data_args(parser)
    parser = runner_name.parse_runner_args(parser)
    parser = model_name.parse_model_args(parser)
    args, extras = parser.parse_known_args()

    # Logging configuration
    log_args = [init_args.model_name, args.dataset, str(args.random_seed)]
    for arg in ['lr', 'l2'] + model_name.extra_log_args:
        log_args.append(arg + '=' + str(eval('args.' + arg)))
    log_file_name = '__'.join(log_args).replace(' ', '__')
    if args.log_file == '':
        args.log_file = '../log/{}/{}.txt'.format(init_args.model_name, log_file_name)
    if args.model_path == '':
        args.model_path = '../model/{}/{}.pt'.format(init_args.model_name, log_file_name)

    utils.check_dir(args.log_file)
    logging.basicConfig(filename=args.log_file, level=args.verbose)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(init_args)

    main()
