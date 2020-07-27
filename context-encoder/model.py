import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, channels=3):
        super(Generator, self).__init__()

        def downsample(in_feat, out_feat, normalize=True):
            layers = [nn.Conv2d(in_feat, out_feat, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2))
            return layers

        def upsample(in_feat, out_feat, normalize=True):
            layers = [nn.ConvTranspose2d(in_feat, out_feat, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_feat, 0.8))
            layers.append(nn.ReLU())
            return layers

        self.model = nn.Sequential(*downsample(channels, 64, normalize=False),
                                   *downsample(64, 64),
                                   *downsample(64, 128),
                                   *downsample(128, 256),
                                   *downsample(256, 512),
                                   nn.Conv2d(512, 4000, 1),
                                   *upsample(4000, 512),
                                   *upsample(512, 256),
                                   *upsample(256, 128),
                                   *upsample(128, 64),
                                   nn.Conv2d(64, channels, 3, 1, 1),
                                   nn.Tanh())

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, channels=3):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, stride, normalize):
            layers = [nn.Conv2d(in_filters, out_filters, 3, stride, 1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(*discriminator_block(channels, 64, 2, False),
                                   *discriminator_block(64, 128, 2, True),
                                   *discriminator_block(128, 256, 2, True),
                                   *discriminator_block(256, 512, 1, True),
                                   nn.Conv2d(512, 1, 3, 1, 1))

    def forward(self, img):
        return self.model(img)
