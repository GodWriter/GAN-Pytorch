import os
import numpy as np

import torch

from torch.autograd import Variable
from torchvision.utils import save_image

from config import parse_args
from dataloader import mnist_loader
from model import Generator, Discriminator


def train():
    os.makedirs("images", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    # get configs and dataloader
    opt = parse_args()
    data_loader = mnist_loader(opt)

    # Initialize generator and discriminator
    generator = Generator(opt)
    discriminator = Discriminator(opt)

    if cuda:
        generator.cuda()
        discriminator.cuda()

    # Optimizers
    optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=opt.lr)
    optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=opt.lr)

    for epoch in range(opt.epochs):
        for i, (imgs, _) in enumerate(data_loader):

            # Configure input
            z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))
            gen_imgs = generator(z)
            real_imgs = Variable(imgs.type(Tensor))

            # ------------------
            # Train Discriminator
            # ------------------

            optimizer_D.zero_grad()
            d_loss = -torch.mean(discriminator(real_imgs)) + torch.mean(discriminator(gen_imgs.detach()))

            d_loss.backward()
            optimizer_D.step()

            # Clip weights of discriminator
            for p in discriminator.parameters():
                p.data.clamp_(-opt.clip_value, opt.clip_value)

            # ------------------
            # Train Generator
            # ------------------

            if i % opt.n_critic == 0:
                optimizer_G.zero_grad()
                g_loss = -torch.mean(discriminator(gen_imgs))

                g_loss.backward()
                optimizer_G.step()

            # ------------------
            # Log Information
            # ------------------

            print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" %
                  (epoch, opt.epochs, i, len(data_loader), d_loss.item(), g_loss.item()))

            batches_done = epoch * len(data_loader) + i
            if batches_done % opt.sample_interval == 0:
                save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)

            if batches_done % opt.checkpoint_interval == 0:
                torch.save(generator.state_dict(), "checkpoints/generator_%d.pth" % epoch)
                # torch.save(discriminator.state_dict(), "checkpoints/discriminator_%d.pth" % epoch)

    torch.save(generator.state_dict(), "checkpoints/generator_done.pth")
    print("Training Process has been Done!")


if __name__ == '__main__':
    train()


