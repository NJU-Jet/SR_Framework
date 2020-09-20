import torch
import torch.nn as nn
import logging
from .blocks import MeanShift
import torch.nn.functional as F

class FeedbackBlock(nn.Module):
    def __init__(self, num_features, num_groups, upscale_factor):
        super(FeedbackBlock, self).__init__()
        if upscale_factor == 2:
            stride = 2
            padding = 2
            kernel = 6
        elif upscale_factor == 3:
            stride = 3
            padding = 2
            kernel = 7
        elif upscale_factor == 4:
            stride = 4
            padding = 2
            kernel = 8
        elif upscale_factor == 8:
            stride = 8
            padding = 2
            kernel = 12

        self.num_groups = num_groups

        self.compress_in = nn.Sequential(
            nn.Conv2d(num_features*2, num_features, 1, 1, 0),
            nn.PReLU(init=0.2)
        )

        self.upBlocks = nn.ModuleList()
        self.downBlocks = nn.ModuleList()
        self.uptranBlocks = nn.ModuleList()
        self.downtranBlocks = nn.ModuleList()

        for idx in range(num_groups):
            if idx > 0:
                self.uptranBlocks.append(
                    nn.Sequential(
                        nn.Conv2d(num_features*(idx+1), num_features, 1, 1, 0),
                        nn.PReLU(init=0.2)
                    )
                )
            self.upBlocks.append(
                nn.Sequential(
                    nn.ConvTranspose2d(num_features, num_features, kernel, stride, padding),
                    nn.PReLU(init=0.2)
                )
            )
            if idx > 0:
                self.downtranBlocks.append(
                    nn.Sequential(
                        nn.Conv2d(num_features*(idx+1), num_features, 1, 1, 0),
                        nn.PReLU(init=0.2)
                    )
                )
            self.downBlocks.append(
                nn.Sequential(
                    nn.Conv2d(num_features, num_features, kernel, stride, padding),
                    nn.PReLU(init=0.2)
                )
            )

        self.compress_out = nn.Sequential(
            nn.Conv2d(num_features*num_groups, num_features, 1, 1, 0),
            nn.PReLU(init=0.2)
        )

        self.first_FB = True
        self.last_hidden = None

    def forward(self, x):
        if self.first_FB:
            self.last_hidden = torch.zeros(x.shape).cuda()
            self.last_hidden.copy_(x)
            self.first_FB = False
    
        x = torch.cat([x, self.last_hidden], dim = 1)
        x = self.compress_in(x)
    
        lr_features = []
        hr_features = []
        lr_features.append(x)

        for idx in range(self.num_groups):
            LD_L = torch.cat(lr_features, dim=1)
            if idx > 0:
                LD_L = self.uptranBlocks[idx-1](LD_L)
            hr_features.append(self.upBlocks[idx](LD_L))
            
            LD_H = torch.cat(hr_features, dim=1)
            if idx > 0:
                LD_H = self.downtranBlocks[idx-1](LD_H)
            lr_features.append(self.downBlocks[idx](LD_H))

        #del hr_features
        output = torch.cat(lr_features[1:], dim=1)
        output = self.compress_out(output)
        
        self.last_hidden = output
        return output

    def reset(self):
        self.first_FB = True


class SRFBN(nn.Module):
    def __init__(self, upscale_factor, in_channels, out_channels, num_fea, num_steps, num_groups, act_type='prelu', norm_type=None):
        super(SRFBN, self).__init__()
        
        if upscale_factor == 2:
            stride = 2
            padding = 2
            kernel = 6
        elif upscale_factor == 3:
            stride = 3
            padding = 2
            kernel = 7
        elif upscale_factor == 4:
            stride = 4
            padding = 2
            kernel = 8
        elif upscale_factor == 8:
            stride = 8
            padding = 2
            kernel = 12

        self.num_steps = num_steps
        self.num_features = num_features
        self.upscale_factor = upscale_factor
        
        # RGB mean for DIV2K
        rgb_mean = [0.4469, 0.4367, 0.4044]
        #rgb_mean = [0.4488, 0.4371, 0.4040]
        rgb_std = [1.0, 1.0, 1.0]
        self.sub_mean = MeanShift(rgb_mean, rgb_std)
    
        # LR feature extraction block
        self.conv_in = nn.Sequential(
            nn.Conv2d(in_channels, 4*num_features, 3, 1, 1),
            nn.PReLU(init=0.2)
        )
        self.feat_in = nn.Sequential(
            nn.Conv2d(4*num_features, num_features, 1, 1, 0),
            nn.PReLU(init=0.2)
        )

        # Feedback block
        self.block = FeedbackBlock(num_features, num_groups, upscale_factor)

        # Reconstruction
        self.out = nn.Sequential(
            nn.ConvTranspose2d(num_features, num_features, kernel, stride, padding),
            nn.PReLU(init=0.2)
        )
        self.conv_out = nn.Sequential(
            nn.Conv2d(num_features, out_channels, 3, 1, 1),
        )
        self.add_mean = MeanShift(rgb_mean, rgb_std, sign=1)
                
    
    def forward(self, x):
        self.block.reset()
                
        x = self.sub_mean(x)
        #inter_res = F.interpolate(x, scale_factor=self.upscale_factor, mode='bilinear', align_corners=False)

        # extract features
        x = self.conv_in(x)
        x = self.feat_in(x)
        return self.conv_out(self.out(x))
        
        outs = []
        for _ in range(self.num_steps):
            h = self.block(x)
            #h = torch.add(inter_res, self.conv_out(self.out(h)))
            h = self.conv_out(self.out(h))
            h = self.add_mean(h)
            outs.append(h)

        return outs
