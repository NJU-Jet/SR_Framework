import argparse
import torch
import random
from options import test_parse
from utils import *
from data import create_dataset, create_loader
from solvers.networks import create_model
from solvers import create_solver
import os
import os.path as osp
import warnings
warnings.filterwarnings('ignore')


def main():
    
    # file and stream logger
    log_path = 'log/logger_info.log'
    lg = logger('Base', log_path)
    pn = 40
    print('\n','-'*pn, 'General INFO', '-'*pn)

    # setting arguments
    parser = argparse.ArgumentParser(description='Test arguments')
    parser.add_argument('--opt', type=str, required=True, help='path to test yaml file')
    parser.add_argument('--dataset_name', type=str, default=None)
    parser.add_argument('--scale', type=int, required=True)

    args = parser.parse_args()
    args = test_parse(args, lg)
    
    # create test dataloader
    test_dataset = create_dataset(args['datasets']['test'])
    test_loader = create_loader(test_dataset, args['datasets']['test'])
    lg.info('\nHR root: [{}]\nLR root: [{}]'.format(args['datasets']['test']['dataroot_HR'], args['datasets']['test']['dataroot_LR']))
    lg.info('Number of test images: [{}]'.format(len(test_dataset)))

    # create model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = create_model(args['networks']).to(device)
    lg.info('Create model: [{}]'.format(args['networks']['which_model']))
    scale = args['scale']
    state_dict = torch.load(args['networks']['pretrained'])
    lg.info('Load pretrained from: [{}]'.format(args['networks']['pretrained']))
    model.load_state_dict(state_dict)
    
    # calculate cuda time
    if args['calc_cuda_time']:
        lg.info('Start calculating cuda time...')
        avg_test_time = calc_cuda_time(test_loader, model)
        lg.info('Average cuda time: [{:.5f}]'.format(avg_test_time))
    
    # Test
    print('\n', '-'*pn, 'Testing {}'.format(args['dataset_name']), '-'*pn)
    #pbar = ProgressBar(len(test_loader))
    psnr_list = []
    ssim_list = []
    time_list = []

    for iter, data in enumerate(test_loader):
        lr = data['LR'].to(device)
        hr = data['HR']
        
        # calculate evaluation metrics
        sr = model(lr)
        psnr, ssim = calc_metrics(tensor2np(sr), tensor2np(hr), crop_border=scale, test_Y=True)
        psnr_list.append(psnr)
        ssim_list.append(ssim)

        #pbar.update('')
        print('[{:03d}/{:03d}] || PSNR/SSIM: {:.2f}/{:.4f} || {}'.format(iter+1, len(test_loader), psnr, ssim, data['filename']))
    
    avg_psnr = sum(psnr_list) / len(psnr_list)
    avg_ssim = sum(ssim_list) / len(ssim_list)

    print('\n','-'*pn, 'Summary', '-'*pn)
    print('Average PSNR: {:.2f}  Average SSIM: {:.4f}'.format(avg_psnr, avg_ssim))


    print('\n','-'*pn, 'Finish', '-'*pn)

def tensor2np(t): # CHW -> HWC, [0, 1] -> [0, 255]
    return (t[0]*255).cpu().clamp(0, 255).round().byte().permute(1, 2, 0).numpy()

if __name__ == '__main__':
    main()
