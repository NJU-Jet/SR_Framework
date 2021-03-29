import json
import yaml
import os
import os.path as osp
import torch
import logging
import sys
sys.path.append('../')
from utils import logger
import shutil

def parse(opt):
    path, name, pretrained = opt.opt, opt.name, opt.pretrained
    
    with open(path, 'r') as fp:        
        args = yaml.full_load(fp.read())
    lg = logger(name, 'log/{}.log'.format(name), pretrained)
        
    # general settings
    args['name'] = name
    args['scale'] = opt.scale
    args['train_Y'] = opt.train_Y
    args['use_chop'] = opt.use_chop

    # setting for datasets
    scale = opt.scale
       
    for phase, dataset_opt in args['datasets'].items():
        dataset_opt['scale'] = scale
        dataset_opt['split'] = phase
        if opt.ps is not None:
            dataset_opt['patch_size'] = opt.ps
        if opt.bs is not None:
            dataset_opt['batch_size'] = opt.bs
        dataset_opt['train_Y'] = opt.train_Y
        if 'XN' in dataset_opt['dataroot_LR']:
            dataset_opt['dataroot_LR'] = dataset_opt['dataroot_LR'].replace('N', str(opt.scale))        

    # setting for networks
    args['networks']['upscale_factor'] = scale
    if opt.train_Y:
        args['networks']['in_channels'] = 1
        args['networks']['out_channels'] = 1

    # setting for solver
    if opt.lr is not None:
        args['solver']['learning_rate'] = opt.lr
    
      
    # setting for GPU environment
    if opt.gpu_ids is not None:
        gpu_list = opt.gpu_ids
    else:
        gpu_list = ','.join([str(x) for x in args['gpu_ids']])
    lg.info('Available gpus: {}'.format(gpu_list))
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list    
    if ',' in gpu_list:
        args['networks']['dataparallel'] = True
        lg.info('Using Dataparallel')

    # create directories for paths
    args['solver']['pretrained'] = pretrained

    root = osp.join(args['paths']['experiment_root'], name)
    args['paths']['root'] = root
    args['paths']['epochs'] = osp.join(root, 'epochs')
    args['paths']['visual'] = osp.join(root, 'visual')
    args['paths']['records'] = osp.join(root, 'records')
    
    if osp.exists(root) and pretrained is None:
        lg.info('Remove dir: [{}]'.format(root))
        shutil.rmtree(root, True)    
    
    for name, path in args['paths'].items():
        if not osp.exists(path):
            os.mkdir(path)
            lg.info('Create directory: {}'.format(path))
 
    return dict_to_nonedict(args), lg


class NoneDict(dict):
    def __missing__(self, key):
        return None


def dict_to_nonedict(opt):
    if isinstance(opt, dict):
        for k, v in opt.items():
            opt[k] = dict_to_nonedict(v)
        return NoneDict(**opt)
    elif isinstance(opt, list):
        return [dict_to_nonedict(x) for x in opt]
    else:
        return opt
