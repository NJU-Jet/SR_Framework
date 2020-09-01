from .SRFBN import SRFBN
from .baseline import model as base_model
from .DHF import DHF
from .IDN import IDN
from .EDSR import EDSR
from .FTN import FTN
from .IMDN import IMDN
from .CARN import CARN
from .LapSRN import LapSRN
from .VDSR import VDSR
from .DRRN import DRRN
from .DRCN import DRCN
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
    elif which_model == 'IDN':
        model = IDN(opt['upscale_factor'], opt['in_channels'], opt['num_fea'], opt['out_channels'])
    elif which_model == 'EDSR':
        model = EDSR(opt['upscale_factor'], opt['in_channels'], opt['num_fea'], opt['out_channels'], opt['n_resblocks'])
    elif which_model == 'FTN':
        model = FTN()
    elif which_model == 'IMDN':
        model = IMDN(opt['upscale_factor'], opt['in_channels'], opt['num_fea'], opt['out_channels'], opt['imdn_blocks'], opt['skip'])
    elif which_model == 'CARN':
        model = CARN(opt['upscale_factor'], opt['in_channels'], opt['num_fea'], opt['out_channels'], opt['use_skip'])
    elif which_model == 'LapSRN':
        model = LapSRN(opt['upscale_factor'], opt['in_channels'], opt['num_fea'], opt['out_channels'])
    elif which_model == 'VDSR':
        model = VDSR(opt['upscale_factor'], opt['in_channels'], opt['num_fea'], opt['out_channels'])
    elif which_model == 'DRRN':
        model = DRRN(opt['upscale_factor'], opt['in_channels'], opt['num_fea'], opt['out_channels'], opt['num_U'])
    elif which_model == 'DRCN':
        model = DRCN(opt['upscale_factor'], opt['in_channels'], opt['num_fea'], opt['out_channels'], opt['recurisive_times'])
    else:
        raise NotImplementedError('unrecognized model: {}'.format(which_model))

    return model
