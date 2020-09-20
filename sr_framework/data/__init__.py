import torch.utils.data as data
import os

def create_dataset(opt):
    if opt['mode'] == 'TrainLRHR':
        from .TrainLRHRDataset import TrainLRHR as D
    elif opt['mode'] == 'TestLR':
        from .TestLRDataset import TestLR as D
    elif opt['mode'] == 'TestLRHRDataset':
        from .TestLRHR import TestLRHR as D
    elif opt['mode'] == 'DIV2K':
        from .DIV2KDataset import DIV2KDataset as D
    else:
        raise NotImplementedError('{} Dataset is not implemented yet!'.format(opt['mode']))
    return D(opt)

def create_loader(dataset, opt):
    if opt['split'] == 'train':
        bs = opt['batch_size']
        shuffle = True
        num_workers = opt['n_workers']
    else:
        bs = 1
        shuffle = False
        num_workers = 1
    
    return data.DataLoader(dataset, batch_size=bs, shuffle=shuffle, num_workers=num_workers, pin_memory=True, drop_last=True)
