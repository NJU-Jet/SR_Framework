from .SRFBN import SRFBN
from .baseline import model as base_model
from .DHF import DHF
import torch
import torch.nn as nn

def create_model(opt):
    which_model = opt['which_model']
    if which_model == 'SRFBN':
        model = SRFBN(in_channels=opt['in_channels'], out_channels=opt['out_channels'], num_fea=opt['num_fea'], num_steps=opt['num_steps'], num_groups=opt['num_groups'], upscale_factor=opt['upscale_factor'])
    elif which_model == 'baseline':
        model = base_model()
    elif which_model == 'DHF':
        model = DHF(opt['upscale_factor'], opt['in_channels'], opt['num_fea'], opt['out_channels'])
    else:
        raise NotImplementedError('unrecognized model: {}'.format(which_model))

    return model
