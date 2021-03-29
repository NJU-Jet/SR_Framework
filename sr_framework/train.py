import argparse
import torch
import random
from options import parse
from utils import *
from data import create_dataset, create_loader
from solvers import create_solver
from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
import os
import os.path as osp
import warnings
warnings.filterwarnings('ignore')

def main():

    # setting arguments and logger
    parser = argparse.ArgumentParser(description='Arguments')
    parser.add_argument('--opt', type=str, required=True, help='path to json or yaml file')
    parser.add_argument('--name', type=str, required=True, help='save_dir prefix name')
    parser.add_argument('--scale', type=int, required=True, help='scale factor')
    parser.add_argument('--ps', type=int, default=64, help='patch size')
    parser.add_argument('--bs', type=int, default=16, help='batch_size')
    parser.add_argument('--lr', type=float, default=None, help='learning rate')
    parser.add_argument('--train_Y', action='store_true', default=False, help='convert rgb to yuv and only train on Y channel')
    parser.add_argument('--gpu_ids', type=str, default=None, help='which gpu to use')
    parser.add_argument('--use_chop', action='store_true', default=False, help='whether to use split_forward in test phase')
    parser.add_argument('--pretrained', default=None, help='checkpoint path to resume from')

    args = parser.parse_args()
    args, lg = parse(args)

    # Tensorboard curve
    pretrained = args['solver']['pretrained']
    tensorboard_path = 'Tensorboard/{}'.format(args['name'])
    
    if pretrained is None:
        if osp.exists(tensorboard_path):
            lg.info('Remove dir: [{}]'.format(tensorboard_path))
            shutil.rmtree(tensorboard_path, True)
            
    writer = SummaryWriter(tensorboard_path)

    # random seed
    seed = args['solver']['manual_seed']
    random.seed(seed)
    torch.manual_seed(seed)
    
    # create train and val dataloader
    for phase, dataset_opt in args['datasets'].items():
        if phase == 'train':
            train_dataset = create_dataset(dataset_opt)
            train_loader = create_loader(train_dataset, dataset_opt)
            length = len(train_dataset)
            lg.info('Number of train images: [{}], iters each epoch: [{}]'.format(length, len(train_loader)))
        elif phase == 'val':
            val_dataset = create_dataset(dataset_opt)
            val_loader = create_loader(val_dataset, dataset_opt)
            length = len(val_dataset)
            lg.info('Number of val images: [{}], iters each epoch: [{}]'.format(length, len(val_loader)))
        elif phase == 'test':
            test_dataset = create_dataset(dataset_opt)
            test_loader = create_loader(test_dataset, dataset_opt)
            length = len(test_dataset)
            lg.info('Number of test images: [{}], iters each epoch: [{}]'.format(length, len(test_loader)))
                
    # create solver
    solver = create_solver(args)

    # training prepare
    solver_log = solver.get_current_log()
    NUM_EPOCH = args['solver']['num_epochs']
    cur_iter = -1
    start_epoch = solver_log['epoch']    
    scale = args['scale']
    lg.info('Start Training from [{}] Epoch'.format(start_epoch))
    print_freq = args['print']['print_freq']
    val_step = args['solver']['val_step']

    # training 
    for epoch in range(start_epoch, NUM_EPOCH+1):
        solver_log['epoch'] = epoch
        
        train_loss_list = []
        for iter, data in enumerate(train_loader):
            cur_iter += 1
            solver.feed_data(data)
            iter_loss = solver.optimize_step()
            train_loss_list.append(iter_loss)
        
            # show on screen
            if (cur_iter % print_freq) == 0:
                lg.info('Epoch: {:4} | iter: {:3} | train_loss: {:.4f} | lr: {}'.format(epoch, iter, iter_loss, solver.get_current_learning_rate()))

        train_loss = round(sum(train_loss_list) / len(train_loss_list), 4)
        writer.add_scalar('X{}/train_loss'.format(args['scale']), train_loss, epoch)
        solver_log['train_records']['train_loss'].append(train_loss)
        solver_log['train_records']['lr'].append(solver.get_current_learning_rate())

        epoch_is_best = False

        if (epoch % val_step) == 0:        
            # Validation
            lg.info('Start Validation...')
            pbar = ProgressBar(len(val_loader))
            psnr_list = []
            ssim_list = []
            val_loss_list = []

            for iter, data in enumerate(val_loader):
                solver.feed_data(data)
                loss = solver.test()
                val_loss_list.append(loss)
        
                # calculate evaluation metrics
                visuals = solver.get_current_visual(need_np=True)
                psnr, ssim = calc_metrics(visuals['SR'], visuals['HR'], crop_border=scale, test_Y=True)
                psnr_list.append(psnr)
                ssim_list.append(ssim)
                pbar.update('')            

            # save image
            solver.save_current_visual(epoch)
        
            avg_psnr = round(sum(psnr_list) / len(psnr_list), 2)
            avg_ssim = round(sum(ssim_list) / len(ssim_list), 4)
            val_loss = round(sum(val_loss_list) / len(val_loss_list), 4)
            writer.add_scalar('X{}/val_loss'.format(args['scale']), val_loss, epoch)
            writer.add_scalar('X{}/psnr'.format(args['scale']), avg_psnr, epoch)
            writer.add_scalar('X{}/ssim'.format(args['scale']), avg_ssim, epoch)

            solver_log['val_records']['val_loss'].append(val_loss)
            solver_log['val_records']['psnr'].append(avg_psnr)
            solver_log['val_records']['ssim'].append(avg_ssim)

            # record the best epoch
            if solver_log['best_pred'] < avg_psnr:
                solver_log['best_pred'] = avg_psnr
                epoch_is_best = True
                solver_log['best_epoch'] = epoch
            lg.info('PSNR: {:.2f} | SSIM: {:.4f} | Loss: {:.4f} | Best_PSNR: {:.2f} in Epoch: [{}]'.format(avg_psnr, avg_ssim, val_loss, solver_log['best_pred'], solver_log['best_epoch']))

        solver.set_current_log(solver_log)
        solver.save_checkpoint(epoch, epoch_is_best)
        solver.save_current_log()

        # update lr
        solver.update_learning_rate(epoch)
    
    lg.info('===> Finished !')


if __name__ == '__main__':
	main()
