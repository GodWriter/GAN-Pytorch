import os
import numpy as np

import torch

from torch.autograd import Variable

from config import parse_args
from model import Generator, Discriminator


def train():
    os.makedirs("images", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    cuda = True if torch.cuda.is_available() else False
    FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

    # get configs and dataloader
    opt = parse_args()
    data_loader = mnist_loader(opt)

    # Initialize generator and discriminator
    generator = Generator(opt)
    discriminator = Discriminator(opt)

    # Loss function
    adversarial_loss = torch.nn.MSELoss()

    if cuda:
        generator.cuda()
        discriminator.cuda()
        adversarial_loss.cuda()

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    for epoch in range(opt.epochs):
        for i, (imgs, labels) in enumerate(data_loader):

            # Adversarial ground truths
            valid = Variable(FloatTensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)
            fake = Variable(FloatTensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)

            # Configure input
            z = Variable(FloatTensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))
            gen_labels = Variable(LongTensor(np.random.randint(0, opt.n_classes, imgs.shape[0])))
            labels = Variable(labels.type(LongTensor))

            real_imgs = Variable(imgs.type(FloatTensor))
            gen_imgs = generator(z, gen_labels)

            # ------------------
            # Train Discriminator
            # ------------------

            optimizer_D.zero_grad()

            real_loss = adversarial_loss(discriminator(real_imgs, labels), valid)
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach(), gen_labels), fake)
            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()

            # ------------------
            # Train Generator
            # ------------------

            if i % opt.n_critic == 0:
                optimizer_G.zero_grad()

                # Loss for generator
                g_loss = adversarial_loss(discriminator(gen_imgs, gen_labels), valid)

                # Update parameters
                g_loss.backward()
                optimizer_G.step()

            # ------------------
            # Log Information
            # ------------------

            print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" %
                  (epoch, opt.epochs, i, len(data_loader), d_loss.item(), g_loss.item()))

            batches_done = epoch * len(data_loader) + i
            if batches_done % opt.sample_interval == 0:
                sample_image(opt, 10, batches_done, generator, FloatTensor, LongTensor)

            if batches_done % opt.checkpoint_interval == 0:
                torch.save(generator.state_dict(), "checkpoints/generator_%d.pth" % epoch)
                # torch.save(discriminator.state_dict(), "checkpoints/discriminator_%d.pth" % epoch)

    torch.save(generator.state_dict(), "checkpoints/generator_done.pth")
    print("Training Process has been Done!")


if __name__ == '__main__':
    train()
