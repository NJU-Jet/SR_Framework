import torch
import sys
import time
import torch.nn as nn
import torch.optim as optim
from .BaseSolver import BaseSolver
from .networks import create_model
from .networks import baseline
import logging
import os
import os.path as osp
import torchvision.utils as thutil
import imageio
import pandas as pd
from shutil import get_terminal_size

class SRSolver(BaseSolver):
    def __init__(self, opt):
        super(SRSolver, self).__init__(opt)
        self.lg = logging.getLogger(opt['name'])
        self.train_opt = opt['solver']
        self.scale = opt['scale']
        self.use_chop = opt['use_chop']
        self.lg.info('use split forward in test phase: [{}]'.format(self.use_chop))

        self.train_records = {
            'train_loss': [],
            'lr': []
        }
        self.val_records = {
            'val_loss': [],
            'psnr': [],
            'ssim': []
        }
        self.val_step = self.train_opt['val_step']
        self.lg.info('Validation step: [{}]'.format(self.val_step))

        # create model
        self.model = create_model(opt['networks'])
        self.model = self.model.to(self.device)
        if opt['networks']['dataparallel']:
            self.lg.info('DataParallel')
            self.model = nn.DataParallel(self.model)
        self.lg.info('Model: [{}]'.format(opt['networks']['which_model'])) 

        # set loss
        loss_type = self.train_opt['loss_type']
        if loss_type == 'l1':
            self.criterion = nn.L1Loss()
        elif loss_type == 'l2':
            self.criterion = nn.MSELoss()
        else:
            raise NotImplementedError('Loss type [{}] is not implemented!'.format(loss_type))
        self.lg.info('Criterion: [{}]'.format(loss_type))

        # set optimizer
        optim_type = self.train_opt['type']
        lr = self.train_opt['learning_rate']
        weight_decay = train_opt['weight_decay']
        if optim_type == 'ADAM':
            self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr, weight_decay=weight_decay)
        elif optim_type == 'SGD':
            self.optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr, momentum=0.9, weight_decay=weight_decay)
        else:
            raise NotImplementedError('Optimizer type [{}] is not implemented!'.format(optim))
        self.lg.info('Optimizer: [{}], lr: [{}]'.format(optim_type, lr))

        # set lr_scheduler
        scheduler = self.train_opt['lr_scheme']
        self.scheduler_name = scheduler
        steps = self.train_opt['lr_steps']
        lr_gamma = self.train_opt['lr_gamma']
        if scheduler == 'MultiStepLR':
            self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, steps, lr_gamma)
        elif scheduler == 'ReduceLROnPlateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5)
        else:
            raise NotImplementedError('lr_scheduler type [{}] is not implemented!'.format(scheduler))
        self.lg.info('Scheduler: [{}], steps: {}, gamma: [{}]'.format(scheduler, steps, lr_gamma)) 

        # Network sumary
        #self.summary_network(self.model)
        num_pra, GF = self.count_parameters(self.model)
        self.lg.info('Total parameters: [{:.3f}M], GFlops: [{:.4f}G]'.format(num_pra / 1e6, GF / 1e9))

        # Init network or load network
        self.load()


    def feed_data(self, data):
        input = torch.zeros(data['LR'].shape)
        self.LR = input.copy_(data['LR']).to(self.device)
        target = torch.zeros(data['HR'].shape)
        self.HR = target.copy_(data['HR']).to(self.device)


    def optimize_step(self):
        self.model.train()
        self.optimizer.zero_grad()

        loss = 0.0
        outputs = self.model(self.LR)
        loss = self.criterion(outputs, self.HR)
        self.loss = loss.item()

        loss.backward()

        # for stable training
        if loss.item() < self.skip_threshold * self.last_epoch_loss:
            self.optimizer.step() 
            self.last_epoch_loss = loss.item()
        else:
            self.lg.warning('Skip this batch! [Loss: {:.4f}]'.format(loss.item()))

        self.model.eval()
        return loss.item()    


    def test(self):
        self.model.eval()
        with torch.no_grad():
            if self.use_chop:
                self.SR = self.split_forward(self.LR)
            else:
                self.SR = self.model(self.LR)
            
        self.model.train()
        loss = self.criterion(self.SR, self.HR)
        return loss.item()


    def split_forward(self, x, overlap_h=10, overlap_w=10, min_size=1e5):
        n_GPUs = 2
        scale = self.scale
        b, c, h, w = x.shape
        h_half, w_half = h // 2, w // 2
    
        # make sure input size can be divided by 2
        if h_half % 2 == 1:
            overlap_h += 1
        if w_half % 2 == 1:
            overlap_w += 1

        h_size, w_size = h_half + overlap_h, w_half + overlap_w

        lr_list = [
            x[:, :, 0:h_size, 0:w_size],
            x[:, :, 0:h_size, (w-w_size):w],
            x[:, :, (h-h_size):h, 0:w_size],
            x[:, :, (h-h_size):h, (w-w_size):w]
        ]

        if w_size * h_size < min_size:
            sr_list = []
            for i in range(0, 4, n_GPUs):
                lr_batch = torch.cat(lr_list[i:(i + n_GPUs)], dim=0)
                sr_batch = self.model(lr_batch)
                sr_list.extend(sr_batch.chunk(n_GPUs, dim=0))
        else:
            sr_list = [self.split_forward(patch) for patch in lr_list]

        h, w = h*scale, w*scale
        h_half, w_half = h_half*scale, w_half*scale
        h_size, w_size = h_size*scale, w_size*scale

        output = x.new(b, c, h, w)
        output[:, :, 0:h_half, 0:w_half] = sr_list[0][:, :, 0:h_half, 0:w_half]
        output[:, :, 0:h_half, w_half:w] = sr_list[1][:, :, 0:h_half, (w_size+w_half-w):w_size]
        output[:, :, h_half:h, 0:w_half] = sr_list[2][:, :, (h_size+h_half-h):h_size, 0:w_half]
        output[:, :, h_half:h, w_half:w] = sr_list[3][:, :, (h_size+h_half-h):h_size, (w_size+w_half-w):w_size]
       
        return output


    def load(self):
        pretrained = self.train_opt['pretrained']
        init_type = self.train_opt['init_type']
        if pretrained is None:
            self.lg.info('Init model weights using [{}] init...'.format(init_type))
            self.init_weight(self.model, init_type=init_type)
        else:
            # resume from checkpoint
            self.lg.info('Resume from [{}]...'.format(pretrained))
            ckp = torch.load(pretrained)
            self.model.load_state_dict(ckp['state_dict'])
            self.cur_epoch = ckp['epoch'] + 1
            self.optimizer.load_state_dict(ckp['optimizer'])
            self.best_pred = ckp['best_pred']
            self.best_epoch = ckp['best_epoch']
            self.train_records = ckp['train_records']
            self.val_records = ckp['val_records']
            self.scheduler.load_state_dict(ckp['lr_scheduler'])

        
    def save_checkpoint(self, epoch, is_best):
        filename = osp.join(self.ckp_path, '{}_ckp.pth'.format(epoch))
        ckp = {
            'state_dict': self.model.state_dict(),
            'epoch': epoch,
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.scheduler.state_dict(),
            'best_pred': self.best_pred,
            'best_epoch': self.best_epoch,
            'train_records': self.train_records,
            'val_records': self.val_records
        }
        if epoch % self.save_ckp_step:
            torch.save(ckp, filename)
            self.lg.info('Saving last checkpoint to [{}]'.format(filename))

        if is_best:
            filename = osp.join(self.ckp_path, 'best.pth')
            torch.save(ckp['state_dict'], filename)
            self.lg.info('Saving best checkpoint to [{}]'.format(filename))


    def get_current_visual(self, need_np):
        out = {}
        # Tensor CHW 
        out['LR'] = (self.LR[0]*255).cpu().clamp(0, 255).round().byte()
        out['SR'] = (self.SR[0]*255).cpu().clamp(0, 255).round().byte()
        out['HR'] = (self.HR[0]*255).cpu().clamp(0, 255).round().byte()

        # Numpy HWC
        if need_np:
            out['LR'] = out['LR'].permute(1, 2, 0).numpy()
            out['SR'] = out['SR'].permute(1, 2, 0).numpy()
            out['HR'] = out['HR'].permute(1, 2, 0).numpy()
                
        return out

   
    def save_current_visual(self, epoch):
        if (epoch % self.save_visual_step) == 0:
            visuals = self.get_current_visual(need_np=False)
            visuals_list = [visuals['SR'], visuals['HR']]
            visual_images = torch.stack(visuals_list)
            visual_images = thutil.make_grid(visual_images, nrow=2, padding=5)
            visual_images = visual_images.permute(1, 2, 0).numpy()
            imageio.imwrite(osp.join(self.visual_path, '{}.png'.format(epoch)), visual_images)


    def get_current_learning_rate(self):
        return self.optimizer.param_groups[0]['lr']

    
    def update_learning_rate(self, epoch):
        if self.scheduler_name == 'ReduceLROnPlateau':
            self.scheduler.step(self.loss)
        else:    
            self.scheduler.step()
       
    
    def get_current_log(self):
        log = {}
        log['epoch'] = self.cur_epoch
        log['best_pred'] = self.best_pred
        log['best_epoch'] = self.best_epoch
        log['train_records'] = self.train_records
        log['val_records'] = self.val_records
        
        return log

    
    def set_current_log(self, log):
        self.cur_epoch = log['epoch']
        self.best_pred = log['best_pred']
        self.best_epoch = log['best_epoch']
        self.train_records = log['train_records']
        self.val_records = log['val_records']
    

    def save_current_log(self):
        train_data_frame = pd.DataFrame(
            data = {'train_loss': self.train_records['train_loss'],
                    'lr': self.train_records['lr']
                    },
                    index=range(0, self.cur_epoch+1)
            )
        train_data_frame.to_csv(osp.join(self.records_path, 'train_records.csv'), sep='\t', index_label='epoch')

        val_data_frame = pd.DataFrame(
            data = {'val_loss': self.val_records['val_loss'],
                    'psnr': self.val_records['psnr'],
                    'ssim': self.val_records['ssim']
                    },
                    index=range(0, self.cur_epoch+1, self.val_step)
            )
        val_data_frame.to_csv(osp.join(self.records_path, 'val_records.csv'), sep='\t', index_label='epoch')

