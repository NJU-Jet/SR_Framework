import torch.nn as nn
import torch


def conv_layer(in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1):
    padding = int((kernel_size - 1) / 2) * dilation
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=True, dilation=dilation,
                     groups=groups)


def activation(act_type, inplace=True, neg_slope=0.05, n_prelu=1):
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'lrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError('activation layer [{:s}] is not found'.format(act_type))
    return layer


def sequential(*args):
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


class ShortcutBlock(nn.Module):
    def __init__(self, submodule):
        super(ShortcutBlock, self).__init__()
        self.sub = submodule

    def forward(self, x):
        output = x + self.sub(x)
        return output


def pixelshuffle_block(in_channels, out_channels, upscale_factor=2, kernel_size=3, stride=1):
    conv = conv_layer(in_channels, out_channels * (upscale_factor ** 2), kernel_size, stride)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    return sequential(conv, pixel_shuffle)


class IMDModule(nn.Module):
    def __init__(self, in_channels, distillation_rate=0.25):
        super(IMDModule, self).__init__()
        self.distilled_channels = int(in_channels * distillation_rate)
        self.remaining_channels = int(in_channels - self.distilled_channels)
        self.c1 = conv_layer(in_channels, in_channels, 3)
        self.c2 = conv_layer(self.remaining_channels, in_channels, 3)
        self.c3 = conv_layer(self.remaining_channels, in_channels, 3)
        self.c4 = conv_layer(self.remaining_channels, self.distilled_channels, 3)
        self.act = activation('lrelu', neg_slope=0.05)
        self.c5 = conv_layer(self.distilled_channels * 4, in_channels, 1)

    def forward(self, input):
        out_c1 = self.act(self.c1(input))
        distilled_c1, remaining_c1 = torch.split(out_c1, (self.distilled_channels, self.remaining_channels), dim=1)
        out_c2 = self.act(self.c2(remaining_c1))
        distilled_c2, remaining_c2 = torch.split(out_c2, (self.distilled_channels, self.remaining_channels), dim=1)
        out_c3 = self.act(self.c3(remaining_c2))
        distilled_c3, remaining_c3 = torch.split(out_c3, (self.distilled_channels, self.remaining_channels), dim=1)
        out_c4 = self.c4(remaining_c3)

        out = torch.cat([distilled_c1, distilled_c2, distilled_c3, out_c4], dim=1)
        out_fused = self.c5(out) + input
        return out_fused


class model(nn.Module):
    def __init__(self, upscale=2, in_nc=3, nf=12, out_nc=3, num_modules=5):
        super(model, self).__init__()

        #fea_conv = [conv_layer(in_nc, nf, kernel_size=3)]
        self.fea_conv = conv_layer(in_nc, nf, kernel_size=3)
        
        #rb_blocks = [IMDModule(in_channels=nf) for _ in range(num_modules)]
        self.rb_blocks = nn.Sequential(
                IMDModule(in_channels=nf),
                IMDModule(in_channels=nf),
                IMDModule(in_channels=nf),
                IMDModule(in_channels=nf),
                IMDModule(in_channels=nf)
        )

        #LR_conv = conv_layer(nf, nf, kernel_size=1)
        self.LR_conv = conv_layer(nf, nf, kernel_size=1)

        #upsample_block = pixelshuffle_block
        #upsampler = upsample_block(nf, out_nc, upscale_factor=upscale)
        self.upsample = nn.Sequential(
                nn.Conv2d(nf, out_nc * (upscale**2), 3, 1, 1),
                nn.PixelShuffle(upscale)
        )

        #self.model1 = sequential(*fea_conv, ShortcutBlock(sequential(*rb_blocks, LR_conv)), *upsampler)

    def forward(self, input):
        x = self.fea_conv(input)

        out = self.rb_blocks(x)
        out = self.LR_conv(out)
        out += x
        output = self.upsample(out)

       #output = (self.model1(input).clamp_(0, 1) * 255.0).round() / 255.0
        return output
