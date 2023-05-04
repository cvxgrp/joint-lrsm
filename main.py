import argparse
import shutil
import logging
import yaml
import numpy as np
from strat_models import *
import os
import torch


def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])

    parser.add_argument('--config',
                        type=str,
                        required=True,
                        help='Path to the config file')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--run',
                        type=str,
                        default='run',
                        help='Path for saving running related data.')
    parser.add_argument('--doc',
                        type=str,
                        default='0',
                        help='A string for documentation purpose')
    parser.add_argument('--comment',
                        type=str,
                        default='',
                        help='A string for experiment comment')

    args = parser.parse_args()

    args.log = os.path.join(args.run, args.doc)

    # parse config file
    if os.path.exists(args.log):
        shutil.rmtree(args.log)
    os.makedirs(args.log)

    with open(os.path.join('configs', args.config), 'r') as f:
        config = yaml.safe_load(f)

    with open(os.path.join(args.log, 'config.yml'), 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    # setup logger
    level = getattr(logging, 'INFO', None)
    handler1 = logging.StreamHandler()
    handler2 = logging.FileHandler(os.path.join(args.log, 'stdout.txt'))
    formatter = logging.Formatter(
        '%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
    handler1.setFormatter(formatter)
    handler2.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    logger.setLevel(level)

    # add device
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    logging.info('Using device: {}'.format(device))

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    return args, config


if __name__ == '__main__':
    args, config = parse_args_and_config()
    logging.info('Writing log file to {}'.format(args.log))
    logging.info('Exp instance id = {}'.format(os.getpid()))
    logging.info('Exp comment = {}'.format(args.comment))
    logging.info('Random seed = {}'.format(args.seed))
    logging.info('Config =')
    print('>' * 80)
    print(config)
    print('<' * 80)

    data_train, data_test, G = get_data_graph(config['dataset'])

    # create model
    loss = eval(config['model']['loss'])(**config['model']['loss_kwargs'])
    reg = eval(config['model']['reg'])(**config['model']['reg_kwargs'])
    bm = BaseModel(loss=loss, reg=reg)
    sm_strat = StratifiedModel(bm, graph=G, config=config['model']['coef'])

    # fitting model
    info = sm_strat.fit(data_train,
                        data_train,
                        **config['optim'],
                        figpath=args.log)
    test_report = sm_strat.report(data_test)
    train_report = sm_strat.report(data_train)

    logging.info(f"Info = {info}")
    logging.info(f"test report = {test_report}")
    logging.info(f"training report = {train_report}")
