## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881

import os, random
import logging
import argparse

import mmcv
import torch
from mmcv.runner import get_dist_info, get_time_str, init_dist
from mmcv import mkdir_or_exist
from basicsr.utils.dist_util import get_dist_info, init_dist
from torchsummaryX import summary
from basicsr.data import create_dataloader, create_dataset
from basicsr.models import create_model
from basicsr.utils.options import dict2str, parse
from basicsr.utils import (get_env_info, get_root_logger, set_random_seed)


def parse_options(is_train=False):
    parser = argparse.ArgumentParser(description='Blind UDC image restoration using Restormer')
    parser.add_argument(
        '-opt', type=str, required=True, help='Path to option YAML file.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    opt = parse(args.opt, is_train=is_train)

    # distributed settings
    if args.launcher == 'none':
        opt['dist'] = False
        print('Disable distributed.', flush=True)
    else:
        opt['dist'] = True
        if args.launcher == 'slurm' and 'dist_params' in opt:
            init_dist(args.launcher, **opt['dist_params'])
        else:
            init_dist(args.launcher)
            print('init dist .. ', args.launcher)

    opt['rank'], opt['world_size'] = get_dist_info()

    # random seed
    seed = opt.get('manual_seed')
    if seed is None:
        seed = random.randint(1, 10000)
        opt['manual_seed'] = seed
    set_random_seed(seed + opt['rank'])

    return opt


def make_exp_dirs(opt):
    """Make dirs for experiments."""
    path_opt = opt['path'].copy()
    if opt['is_train']:
        mkdir_and_rename(path_opt.pop('experiments_root'))
    else:
        mkdir_and_rename(path_opt.pop('results_root'))
    path_opt.pop('strict_load')


def mkdir_and_rename(path):
    """mkdirs. If path exists, rename it with timestamp and create a new one.

    Args:
        path (str): Folder path.
    """
    if os.path.exists(path):
        new_name = path + '_archived_' + get_time_str()
        print(f'Path already exists. Rename it to {new_name}', flush=True)
        os.rename(path, new_name)
    mmcv.mkdir_or_exist(path)


opt = parse_options(is_train=False)
make_exp_dirs(opt)
log_file = os.path.join(opt['path']['log'], f"test_{opt['name']}_{get_time_str()}.log")
logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
logger.info(get_env_info())
logger.info(dict2str(opt))


torch.backends.cudnn.benchmark = True
# create test dataset and dataloader
test_loaders = []
for phase, dataset_opt in sorted(opt['datasets'].items()):
    test_set = create_dataset(dataset_opt)
    test_loader = create_dataloader(test_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=None,
                                    seed=opt['manual_seed'])
    logger.info(f"Number of test images in {dataset_opt['name']}: {len(test_set)}")
    test_loaders.append(test_loader)

##########################
model = create_model(opt)

for test_loader in test_loaders:
    test_set_name = test_loader.dataset.opt['name']
    mkdir_or_exist(os.path.join(opt['path']['visualization'], test_set_name))

    logger.info(f'Testing {test_set_name}...')
    rgb2bgr = opt['val'].get('rgb2bgr', True)
    use_image = opt['val'].get('use_image', True)
    model.validation(test_loader, opt['name'], None, opt['val']['save_img'], rgb2bgr, False)
