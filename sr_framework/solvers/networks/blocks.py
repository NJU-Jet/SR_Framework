import torch
import torch.nn as nn
import sys

class MeanShift(nn.Conv2d):
    def __init__(self, mean, std, sign=-1):
        super(MeanShift, self).__init__(3, 3, 1)
        std = torch.Tensor(std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * torch.Tensor(mean)
        self.bias.data.div_(std)
        for p in self.parameters():
            p.requires_grad = False


class CA(nn.Module):
    def __init__(self, in_channels, ratio=4):
        super(CA, self).__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.max = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_channels, in_channels//ratio, 1, 1, 0, bias=False)
        self.fc2 = nn.Conv2d(in_channels//ratio, in_channels, 1, 1, 0, bias=False)
        
        self.act = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_value = self.avg(x)
        max_value = self.max(x)

        avg_out = self.fc2(self.act(self.fc1(avg_value)))
        max_out = self.fc2(self.act(self.fc1(max_value)))
        
        out = avg_out + max_out
        
        return self.sigmoid(out)


