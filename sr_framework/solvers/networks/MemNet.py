import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummaryX import summary


class MemNet(nn.Module):
    def __init__(self, upscale_factor, in_channels=3, num_fea=64, out_channels=3, num_memblock=6, num_resblock=6):
        super(MemNet, self).__init__()
        #self.feature_extractor = BNReLUConv(in_channels, num_fea)
        self.feature_extractor = nn.Conv2d(in_channels, num_fea, 3, 1, 1)

        self.reconstructor = nn.Sequential(
            nn.Conv2d(num_fea, out_channels*(upscale_factor**2), 3 ,1, 1),
            nn.PixelShuffle(upscale_factor)
        )

        #self.reconstructor = BNReLUConv(channels, in_channels)
        self.dense_memory = nn.ModuleList(
            [MemoryBlock(num_fea, num_resblock, i+1) for i in range(num_memblock)]
        )

    def forward(self, x):
        # x = x.contiguous()
        out = self.feature_extractor(x)
        residual = out

        ys = [out]
        for memory_block in self.dense_memory:
            out = memory_block(out, ys)
        out = self.reconstructor(out+residual)
        
        return out


class MemoryBlock(nn.Module):
    """Note: num_memblock denotes the number of MemoryBlock currently"""
    def __init__(self, channels, num_resblock, num_memblock):
        super(MemoryBlock, self).__init__()
        self.num_resblock = num_resblock
        self.recursive_unit =  ResidualBlock(channels)
        
        self.gate_unit = BNReLUConv((num_resblock+num_memblock) * channels, channels, 1, 1, 0)

    def forward(self, x, ys):
        """ys is a list which contains long-term memory coming from previous memory block
        xs denotes the short-term memory coming from recursive unit
        """
        xs = []
        residual = x
        for i in range(self.num_resblock):
            x = self.recursive_unit(x)
            xs.append(x)
        
        gate_out = self.gate_unit(torch.cat(xs+ys, 1))
        ys.append(gate_out)
        return gate_out


class ResidualBlock(torch.nn.Module):
    """ResidualBlock
    introduced in: https://arxiv.org/abs/1512.03385
    x - Relu - Conv - Relu - Conv - x
    """

    def __init__(self, channels, k=3, s=1, p=1):
        super(ResidualBlock, self).__init__()
        self.relu_conv1 = BNReLUConv(channels, channels, k, s, p)
        self.relu_conv2 = BNReLUConv(channels, channels, k, s, p)
        
    def forward(self, x):
        residual = x
        out = self.relu_conv1(x)
        out = self.relu_conv2(out)
        out = out + residual
        return out


class BNReLUConv(nn.Sequential):
    def __init__(self, in_channels, channels, k=3, s=1, p=1, inplace=True):
        super(BNReLUConv, self).__init__()
        self.add_module('bn', nn.BatchNorm2d(in_channels))
        self.add_module('relu', nn.ReLU(inplace=inplace))
        self.add_module('conv', nn.Conv2d(in_channels, channels, k, s, p, bias=False))

if __name__ == '__main__':
    # 1112K, 90.9G
    s = 4
    in_ = torch.randn(1, 3, round(720/s), round(1280/s)).to('cuda')
    m = MemNet(upscale_factor=s).to('cuda')
    summary(m, in_)
