import os
import torch
import numpy as np

from config import parse_args
from utils import save_sample
from dataloader import celeba_loader
from torch.autograd import Variable
from model import Generator, Discriminator


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def train():
    os.makedirs("images", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    opt = parse_args()
    cuda = True if torch.cuda.is_available() else False
    FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    # Calculate output of image discriminator (PatchGAN)
    patch_h, patch_w = int(opt.mask_size / 2 ** 3), int(opt.mask_size / 2 ** 3)
    patch = (1, patch_h, patch_w)

    # get dataloader
    train_loader = celeba_loader(opt, mode='train')
    test_loader = celeba_loader(opt, mode='test')

    # Initialize generator and discriminator
    generator = Generator(opt.channels)
    discriminator = Discriminator(opt.channels)

    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

    # Loss function
    adversarial_loss = torch.nn.MSELoss()
    pixelwise_loss = torch.nn.L1Loss()

    if cuda:
        generator.cuda()
        discriminator.cuda()
        adversarial_loss.cuda()
        pixelwise_loss.cuda()

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    for epoch in range(opt.epochs):
        for i, (imgs, masked_imgs, masked_parts) in enumerate(train_loader):

            # Adversarial ground truths
            valid = Variable(FloatTensor(imgs.shape[0], *patch).fill_(1.0), requires_grad=False)
            fake = Variable(FloatTensor(imgs.shape[0], *patch).fill_(0.0), requires_grad=False)

            # Configure input
            imgs = Variable(imgs.type(FloatTensor))
            masked_imgs = Variable(masked_imgs.type(FloatTensor))
            masked_parts = Variable(masked_parts.type(FloatTensor))
            gen_parts = generator(masked_imgs)

            # ------------------
            # Train Discriminator
            # ------------------

            optimizer_D.zero_grad()

            real_loss = adversarial_loss(discriminator(masked_parts), valid)
            fake_loss = adversarial_loss(discriminator(gen_parts.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()

            # ------------------
            # Train Generator
            # ------------------

            if i % opt.n_critic == 0:
                optimizer_G.zero_grad()

                # Loss for generator
                g_adv = adversarial_loss(discriminator(gen_parts), valid)
                g_pixel = pixelwise_loss(gen_parts, masked_parts)

                g_loss = 0.001 * g_adv + 0.999 * g_pixel

                # Update parameters
                g_loss.backward()
                optimizer_G.step()

            # ------------------
            # Log Information
            # ------------------

            print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G adv: %f, pixel: %f]" %
                  (epoch, opt.epochs, i, len(train_loader), d_loss.item(), g_adv.item(), g_pixel.item()))

            batches_done = epoch * len(train_loader) + i
            if batches_done % opt.sample_interval == 0:
                save_sample(opt, test_loader, batches_done, generator, FloatTensor)

            if batches_done % opt.checkpoint_interval == 0:
                torch.save(generator.state_dict(), "checkpoints/generator_%d.pth" % epoch)
                # torch.save(discriminator.state_dict(), "checkpoints/discriminator_%d.pth" % epoch)

    torch.save(generator.state_dict(), "checkpoints/generator_done.pth")
    print("Training Process has been Done!")


if __name__ == '__main__':
    train()
