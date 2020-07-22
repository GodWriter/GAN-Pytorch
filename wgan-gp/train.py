import os
import numpy as np

import torch
import torch.autograd as autograd

from torch.autograd import Variable
from torchvision.utils import save_image

from config import parse_args
from dataloader import mnist_loader
from model import Generator, Discriminator


def compute_gradient_penalty(discriminator, real_imgs, gen_imgs, Tensor):
    epsilon = Tensor(np.random.random((real_imgs.size(0), 1, 1, 1)))
    interpolates = (epsilon * real_imgs + ((1 - epsilon) * gen_imgs)).requires_grad_(True)
    fake = Variable(Tensor(real_imgs.shape[0], 1).fill_(1.0), requires_grad=False)

    gradients = autograd.grad(outputs=discriminator(interpolates),
                              inputs=interpolates,
                              grad_outputs=fake,
                              create_graph=True,
                              retain_graph=True,
                              only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

    return gradient_penalty


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
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

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
            gradient_penalty = compute_gradient_penalty(discriminator, real_imgs.data, gen_imgs.data, Tensor)
            d_loss = torch.mean(discriminator(gen_imgs)) - torch.mean(discriminator(real_imgs)) + opt.lambda_gp * gradient_penalty

            d_loss.backward()
            optimizer_D.step()

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


