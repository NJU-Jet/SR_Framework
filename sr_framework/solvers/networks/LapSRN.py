import torch
import torch.nn as nn

class FeatureExtraction(nn.Module):
    def __init__(self, upscale_factor, in_channels=3, num_fea=64):
        super(FeatureExtraction, self).__init__()

        assert (upscale_factor == 2 or upscale_factor == 3), 'Unrecognized scale: [{}]'.format(upscale_factor)
        if upscale_factor == 2:
            stride = 2
            padding = 1
            kernel = 4
        elif upscale_factor == 3:
            stride = 3
            padding = 1
            kernel = 5

        self.conv0 = nn.Conv2d(in_channels, num_fea, 3, 1, 1)

        fea_conv = []
        for i in range(5):
            fea_conv.append(nn.Conv2d(num_fea, num_fea, 3, 1, 1))
        self.fea_conv = nn.Sequential(*fea_conv)

        self.upconv = nn.ConvTranspose2d(num_fea, num_fea, kernel, stride, padding)
        self.act = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        out = self.act(self.conv0(x))
        out = self.act(self.fea_conv(out))
        out = self.act(self.upconv(out))

        return out


class ImageReconstruction(nn.Module):
    def __init__(self, upscale_factor, num_fea, out_channels):
        super(ImageReconstruction, self).__init__()

        assert (upscale_factor == 2 or upscale_factor == 3), 'Unrecognized scale: [{}]'.format(upscale_factor)
        if upscale_factor == 2:
            stride = 2
            padding = 1
            kernel = 4
        elif upscale_factor == 3:
            stride = 3
            padding = 1
            kernel = 5

        self.conv_H = nn.Conv2d(num_fea, out_channels, 3, 1, 1)
        self.conv_L = nn.ConvTranspose2d(out_channels, out_channels, kernel, stride, padding)

    def forward(self, LR, HR_res):
        L = self.conv_L(LR)
        H = self.conv_H(HR_res)

        return L+H


class LapSRN(nn.Module):
    def __init__(self, upscale_factor, in_channels=3, num_fea=64, out_channels=3):
        super(LapSRN, self).__init__()
        upscale_factor_level = 2 if upscale_factor==4 else upscale_factor

        self.feature_extraction1 = FeatureExtraction(upscale_factor_level, in_channels, num_fea)
        self.image_reconstruction1 = ImageReconstruction(upscale_factor_level, num_fea, out_channels)
        
        if upscale_factor == 4:
            self.feature_extraction2 = FeatureExtraction(upscale_factor_level, num_fea, num_fea)
            self.image_reconstruction2 = ImageReconstruction(upscale_factor_level, num_fea, out_channels)

        self.upscale_factor = upscale_factor

    def forward(self, x):
        h1 = self.feature_extraction1(x)
        image = self.image_reconstruction1(x, h1)

        if self.upscale_factor == 4:
            h2 = self.feature_extraction2(h1)
            image = self.image_reconstruction2(image, h2)

        return image
