import json
import yaml
import os
import os.path as osp
import torch
import logging
import sys
import shutil

def parse(opt, lg):
    path, dataset_name, scale = opt.opt, opt.dataset_name, opt.scale
    with open(path, 'r') as fp:        
        args = yaml.full_load(fp.read())
    
    # setting for datasets
    args['scale'] = scale
    args['dataset_name'] = dataset_name
       
    for phase, dataset_opt in args['datasets'].items():
        dataset_opt['scale'] = scale
        dataset_opt['split'] = phase
        if dataset_name is not None:
            dataset_opt['dataroot_HR'] = dataset_opt['dataroot_HR'].replace('dataset_name', dataset_name)
            dataset_opt['dataroot_LR'] = (dataset_opt['dataroot_LR'].replace('dataset_name', dataset_name)).replace('N', str(scale))

    # setting for networks
    args['networks']['upscale_factor'] = scale
      
    # setting for GPU environment
    if args['gpu_ids'] is None:
        gpu_list = ''
    else:
        gpu_list = ','.join([str(x) for x in args['gpu_ids']])
    lg.info('Available gpus: {}'.format(gpu_list))
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list    

    return dict_to_nonedict(args)


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
