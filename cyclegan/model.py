import torch
import torch.nn as nn


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__int__()

        self.block = nn.Sequential(nn.ReflectionPad2d(1),
                                   nn.Conv2d(in_channels, in_channels, 3),
                                   nn.InstanceNorm2d(in_channels),
                                   nn.ReLU(inplace=True),
                                   nn.ReflectionPad2d(1),
                                   nn.Conv2d(in_channels, in_channels, 3),
                                   nn.InstanceNorm2d(in_channels))

    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()

        # Initial convolution block
        out_channels = 64
        model = [nn.ReflectionPad2d(opt.channels),
                 nn.Conv2d(opt.channels, out_channels, 7),
                 nn.InstanceNorm2d(out_channels),
                 nn.ReLU(inplace=True)]
        in_channels = out_channels

        # Downsampling
        for _ in range(2):
            out_channels *= 2
            model += [nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1),
                      nn.InstanceNorm2d(out_channels),
                      nn.ReLU(inplace=True)]
            in_channels = out_channels

        # Residual blocks
        for _ in range(opt.num_residual_blocks):
            model += [ResidualBlock(out_channels)]

        # Upsampling
        for _ in range(2):
            out_channels //= 2
            model += [nn.Upsample(scale_factor=2),
                      nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1),
                      nn.InstanceNorm2d(out_channels),
                      nn.ReLU(inplace=True)]
            in_channels = out_channels

        # Output layer
        model += [nn.ReflectionPad2d(opt.channels),
                  nn.Conv2d(out_channels, opt.channels, 7),
                  nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, opt):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, stride=2, normalize=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride, 1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(*discriminator_block(opt.channels, 64, normalize=False),
                                   *discriminator_block(64, 128),
                                   *discriminator_block(128, 256),
                                   *discriminator_block(256, 512),
                                   nn.ZeroPad2d((1, 0, 1, 0)),
                                   nn.Conv2d(512, 1, 4, padding=1))

    def forward(self, img):
        return self.model(img)
