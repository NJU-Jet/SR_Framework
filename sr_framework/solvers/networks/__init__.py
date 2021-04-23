from .SRFBN import SRFBN
from .IDN import IDN
from .EDSR import EDSR
from .IMDN import IMDN
from .CARN import CARN
from .LapSRN import LapSRN
from .VDSR import VDSR
from .DRRN import DRRN
from .DRCN import DRCN
from .LatticeNet import LatticeNet
from .FSRCNN import FSRCNN
from .RCAN import RCAN
from .MemNet import MemNet

def create_model(opt):
    which_model = opt['which_model']
    if which_model == 'SRFBN':
        model = SRFBN(in_channels=opt['in_channels'], out_channels=opt['out_channels'], num_fea=opt['num_fea'], num_steps=opt['num_steps'], num_groups=opt['num_groups'], upscale_factor=opt['upscale_factor'])
    elif which_model == 'IDN':
        model = IDN(opt['upscale_factor'], opt['in_channels'], opt['num_fea'], opt['out_channels'])
    elif which_model == 'LatticeNet':
        model = LatticeNet(opt['upscale_factor'], opt['in_channels'], opt['num_fea'], opt['out_channels'], opt['num_LBs'])
    elif which_model == 'EDSR':
        model = EDSR(opt['upscale_factor'], opt['in_channels'], opt['num_fea'], opt['out_channels'], opt['n_resblocks'])
    elif which_model == 'IMDN':
        model = IMDN(opt['upscale_factor'], opt['in_channels'], opt['num_fea'], opt['out_channels'], opt['imdn_blocks'])
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
    elif which_model == 'FSRCNN':
        model = FSRCNN(opt['upscale_factor'], opt['in_channels'], opt['out_channels'], opt['d'], opt['s'], opt['m'])
    elif which_model == 'RCAN':
        model = RCAN(opt['upscale_factor'], opt['in_channels'], opt['num_fea'], opt['out_channels'], opt['num_RGs'], opt['num_RCABs'])
    elif which_model == 'MemNet':
        model = MemNet(opt['upscale_factor'], opt['in_channels'], opt['num_fea'], opt['out_channels'], opt['num_memblock'], opt['num_resblock'])
    else:
        raise NotImplementedError('unrecognized model: {}'.format(which_model))

    return model
