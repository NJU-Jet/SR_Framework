import torch
import torch.nn as nn
import torch.nn.functional as F


class D(nn.Module):
    def __init__(self):
        super(D, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(64, 48, 3, 1, 1),
            nn.LeakyReLU(0.05)
        )

        self.group1 = nn.Sequential(
            nn.Conv2d(48, 32, 3, 1, 1, groups=4),
            nn.LeakyReLU(0.05)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.LeakyReLU(0.05)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(48, 64, 3, 1, 1),
            nn.LeakyReLU(0.05)
        )        

        self.group2 = nn.Sequential(
            nn.Conv2d(64, 48, 3, 1, 1, groups=4),
            nn.LeakyReLU(0.05)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(48, 80, 3, 1, 1),
            nn.LeakyReLU(0.05)
        )

        self.last_conv = nn.Sequential(
            nn.Conv2d(80, 64, 1, 1, 0),
            nn.LeakyReLU(0.05)
        )        


    def forward(self, x):
        tmp = self.conv1(x)
        tmp = self.group1(tmp)
        tmp = self.conv2(tmp)
        
        x1, x2 = torch.split(tmp, [16, 48], dim=1)

        x2 = self.conv3(x2)
        x2 = self.group2(x2)
        x2 = self.conv4(x2)

        output = torch.cat([x, x1], dim=1) + x2
        
        output = self.last_conv(output)

        return output


class IDN(nn.Module):
    def __init__(self, upscale_factor=2, in_channels=3, num_fea=64, out_channels=3):
        super(IDN, self).__init__()

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

        self.upscale_factor = upscale_factor 
       
        # extract features
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, 1, 1),
            nn.LeakyReLU(0.05)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.LeakyReLU(0.05)
        )

    
        # DBlock
        self.D = nn.Sequential(
            D(),
            D(),
            D(),
            D(),
        )

        # Reconstruct
        self.upsample = nn.ConvTranspose2d(num_fea, out_channels, kernel, stride, padding) 
        '''
        self.upsample = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.Conv2d(64, out_channels * (upscale_factor ** 2), 3, 1, 1),
            nn.PixelShuffle(upscale_factor)

        )
        '''

    def forward(self, x):
        inter_res = F.interpolate(x, scale_factor=self.upscale_factor, mode='bicubic', align_corners=False)

        # extract features
        x = self.conv1(x)
        x = self.conv2(x)

        # DBlock
        x = self.D(x)
        
        # Reconstruct
        out = self.upsample(x)

        return out + inter_res        
