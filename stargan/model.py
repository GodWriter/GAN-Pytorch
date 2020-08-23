import torch
import torch.nn as nn


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(nn.ReflectionPad2d(1),
                                   nn.Conv2d(in_channels, in_channels, 3, bias=False),
                                   nn.InstanceNorm2d(in_channels, affine=True, track_running_stats=True),
                                   nn.ReLU(inplace=True),
                                   nn.ReflectionPad2d(1),
                                   nn.Conv2d(in_channels, in_channels, 3, bias=False),
                                   nn.InstanceNorm2d(in_channels, affine=True, track_running_stats=True))

    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    def __init__(self, channels=3, res_blocks=9, c_dim=5):
        super(Generator, self).__init__()

        # Initial convolution block
        model = [nn.Conv2d(channels + c_dim, 64, 7, stride=1, padding=3, bias=False),
                 nn.InstanceNorm2d(64, affine=True, track_running_stats=True),
                 nn.ReLU(inplace=True)]

        # Downsampling
        dim = 64
        for _ in range(2):
            model += [nn.Conv2d(dim, dim * 2, 4, stride=2, padding=1, bias=False),
                      nn.InstanceNorm2d(dim * 2, affine=True, track_running_stats=True),
                      nn.ReLU(inplace=True)]
            dim *= 2

        # Residual blocks
        for _ in range(res_blocks):
            model += [ResidualBlock(dim)]

        # Usampling
        for _ in range(2):
            model += [nn.ConvTranspose2d(dim, dim // 2, 4, stride=2, padding=1, bias=False),
                      nn.InstanceNorm2d(dim // 2, affine=True, track_running_stats=True),
                      nn.ReLU(inplace=True)]
            dim = dim // 2

        # Output layer
        model += [nn.Conv2d(dim, channels, 7, stride=1, padding=3), nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, x, c):
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat((x, c), 1)

        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, channel, img_size, c_dim=5, n_strided=6):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, stride=2):
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride, 1),
                      nn.LeakyReLU(0.01, inplace=True)]
            return layers

        dim = 64
        kernel_size = img_size // (2 ** n_strided)

        layers = discriminator_block(channel, dim)
        for _ in range(n_strided - 1):
            layers.extend(discriminator_block(dim, dim * 2))
            dim *= 2

        self.model = nn.Sequential(*layers)
        self.out1 = nn.Conv2d(dim, 1, 3, padding=1, bias=False)
        self.out2 = nn.Conv2d(dim, c_dim, kernel_size, bias=False)

    def forward(self, img):
        feature = self.model(img)

        out_adv = self.out1(feature)
        out_cls = self.out2(feature)

        return out_adv, out_cls.view(out_cls.size(0), -1)
