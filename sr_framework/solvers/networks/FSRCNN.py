import torch
import torch.nn as nn
from torchsummaryX import summary

class FSRCNN(nn.Module):
    def __init__(self, upscale_factor=2, in_channels=3, out_channels=3, d=56, s=12, m=4):
        super(FSRCNN, self).__init__()
        self.m = m

        # Shallow feature extraction
        self.fea_conv = nn.Conv2d(in_channels, d, 5, 1, 2)
        
        # Shrink
        self.shrink_conv = nn.Conv2d(d, s, 1, 1, 0)

        # Mapping
        deep_conv = []
        for i in range(m):
            deep_conv.append(nn.Conv2d(s, s, 3, 1, 1))
        self.deep_convs = nn.ModuleList(deep_conv)

        # Expanding
        self.expand_conv = nn.Conv2d(s, d, 1, 1, 0)

        # Change TransposeConv to PixelShuffle
        self.upsample = nn.Sequential(
            nn.Conv2d(d, out_channels*(upscale_factor)**2, 3, 1, 1),
            nn.PixelShuffle(upscale_factor)
        )

        self.act = nn.ReLU(True)

    def forward(self, x):
        out = self.act(self.fea_conv(x))
        out = self.act(self.shrink_conv(out))
        
        for i in range(self.m):
            out = self.act(self.deep_convs[i](out))

        out = self.act(self.expand_conv(out))

        out = self.upsample(out)
        
        return out

if __name__ == '__main__':
    # 35.14K, 2.0G
    s = 4
    model = FSRCNN(upscale_factor=s).to('cuda')
    in_ = torch.randn(1, 3, round(720/s), round(1280/s)).to('cuda')
    summary(model, in_)
