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
    path, name = opt.opt, opt.name
    
    with open(path, 'r') as fp:        
        args = yaml.full_load(fp.read())
    lg = logger(name, 'log/{}.log'.format(name), args['solver']['pretrained'])
    
    # setting for datasets
    scale = args['scale']
    ps = args['patch_size']    
       
    for phase, dataset_opt in args['datasets'].items():
        dataset_opt['scale'] = scale
        dataset_opt['split'] = phase
        dataset_opt['patch_size'] = ps

    # setting for networks
    args['networks']['upscale_factor'] = scale
      
    # setting for GPU environment
    if args['gpu_ids'] is None:
        gpu_list = ''
    else:
        gpu_list = ','.join([str(x) for x in args['gpu_ids']])
    lg.info('Available gpus: {}'.format(gpu_list))
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list    

    # create directories for paths
    args['name'] = name
    pretrained = args['solver']['pretrained']

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
