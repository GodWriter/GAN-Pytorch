import torch

import numpy as np
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()

        self.img_shape = (opt.channels, opt.img_size, opt.img_size)
        self.label_embedding = nn.Embedding(opt.n_classes, opt.n_classes)

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(*block(opt.latent_dim + opt.n_classes, 128, normalize=False),
                                   *block(128, 256),
                                   *block(256, 512),
                                   *block(512, 1024),
                                   nn.Linear(1024, int(np.prod(self.img_shape))),
                                   nn.Tanh())

    def forward(self, z, labels):
        gen_input = torch.cat((z, self.label_embedding(labels)), -1)
        img = self.model(gen_input)
        img = img.view(img.size(0), *self.img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self, opt):
        super(Discriminator, self).__init__()

        self.img_shape = (opt.channels, opt.img_size, opt.img_size)
        self.label_embedding = nn.Embedding(opt.n_classes, opt.n_classes)

        self.model = nn.Sequential(nn.Linear(opt.n_classes + int(np.prod(self.img_shape)), 512),
                                   nn.LeakyReLU(0.2, inplace=True),
                                   nn.Linear(512, 512),
                                   nn.Dropout(0.4),
                                   nn.LeakyReLU(0.2, inplace=True),
                                   nn.Linear(512, 512),
                                   nn.Dropout(0.4),
                                   nn.LeakyReLU(0.2, inplace=True),
                                   nn.Linear(512, 1))

    def forward(self, img, labels):
        d_in = torch.cat((img.view(img.size(0), -1), self.label_embedding(labels)), -1)
        validity = self.model(d_in)
        return validity

