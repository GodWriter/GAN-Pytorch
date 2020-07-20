import os
import numpy as np
import math

from torchvision.utils import save_image

from torch.autograd import Variable

import torch
import torch.nn as nn
import torch.nn.functional as F


from .config import parse_args
from .dataloader import mnist_loader
from .model import Generator, Discriminator


def train():
    os.makedirs("images", exist_ok=True)
    cuda = True if torch.cuda.is_available() else False

    # get configs and dataloader
    opt = parse_args()
    data_loader = mnist_loader(opt)

    # Initialize generator and discriminator
    generator = Generator(opt)
    discriminator = Discriminator(opt)

    # Loss function
    adversarial_loss = torch.nn.BCELoss()

    if cuda:
        generator.cuda()
        discriminator.cuda()
        adversarial_loss.cuda()

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))


    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    for epoch in range(opt.epochs):
        for i, (imgs, _) in enumerate(data_loader):

            # Adversarial ground truths
            valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)

            # Configure input
            real_imgs = Variable(imgs.type(Tensor))

            pass

    print("Training Done!")


if __name__ == '__main__':
    train()


