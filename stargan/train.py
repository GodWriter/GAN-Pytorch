import os
import time
import torch
import datetime

import torch.nn.functional as F
import torch.autograd as autograd
import numpy as np

from torch.autograd import Variable
from config import parse_args
from utils import save_sample
from model import Generator, Discriminator, weights_init_normal
from dataloader import celeba_loader


def criterion_cls(logit, target):
    return F.binary_cross_entropy_with_logits(logit, target, size_average=False) / logit.size(0)


def compute_gradient_penalty(D, real_samples, fake_samples, FloatTensor):
    alpha = FloatTensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates, _ = D(interpolates)
    fake = Variable(FloatTensor(np.ones(d_interpolates.shape)), requires_grad=False)

    gradients = autograd.grad(outputs=d_interpolates,
                              inputs=interpolates,
                              grad_outputs=fake,
                              create_graph=True,
                              retain_graph=True,
                              only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

    return gradient_penalty


def train():
    opt = parse_args()

    os.makedirs("images/%s" % (opt.dataset), exist_ok=True)
    os.makedirs("checkpoints/%s" % (opt.dataset), exist_ok=True)

    cuda = True if torch.cuda.is_available() else False
    FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    # get dataloader
    train_loader = celeba_loader(opt, mode='train')
    val_loader = celeba_loader(opt, mode='val')

    # Dimensionality
    c_dim = len(opt.selected_attrs)

    # Initialize generator and discriminator
    generator = Generator(opt.channels, opt.residual_blocks, c_dim)
    discriminator = Discriminator(opt.channels, opt.img_height, c_dim)

    # Initialize weights
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

    # Loss function
    cycle_loss = torch.nn.L1Loss()

    if cuda:
        generator = generator.cuda()
        discriminator = discriminator.cuda()
        cycle_loss.cuda()

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    # ------------
    #  Training
    # ------------

    prev_time = time.time()
    for epoch in range(opt.epochs):
        for i, (imgs, labels) in enumerate(train_loader):

            # Model inputs
            imgs = Variable(imgs.type(FloatTensor))
            labels = Variable(labels.type(FloatTensor))

            # Sample label as generator inputs and Generate fake batch of images
            sampled_c = Variable(FloatTensor(np.random.randint(0, 2, (imgs.size(0), c_dim))))
            fake_imgs = generator(imgs, sampled_c)

            # ----------------------
            # Train Discriminator
            # ----------------------

            optimizer_D.zero_grad()

            real_validity, pred_cls = discriminator(imgs)
            fake_validity, _ = discriminator(fake_imgs.detach())
            gradient_penalty = compute_gradient_penalty(discriminator, imgs.data, fake_imgs.data, FloatTensor)

            d_adv_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + opt.lambda_gp * gradient_penalty
            d_cls_loss = criterion_cls(pred_cls, labels)
            D_loss = d_adv_loss + opt.lambda_cls * d_cls_loss

            D_loss.backward()
            optimizer_D.step()

            # -----------------------------
            # Train Generators
            # -----------------------------
            optimizer_G.zero_grad()

            if i % opt.n_critic == 0:
                gen_imgs = generator(imgs, sampled_c)
                recov_imgs = generator(gen_imgs, labels)

                fake_validity, pred_cls = discriminator(gen_imgs)

                g_adv_loss = -torch.mean(fake_validity)
                g_cls_loss = criterion_cls(pred_cls, sampled_c)
                g_rec_loss = cycle_loss(recov_imgs, imgs)
                G_loss = g_adv_loss + opt.lambda_cls * g_cls_loss + opt.lambda_rec * g_rec_loss

                G_loss.backward()
                optimizer_G.step()

                # ------------------
                # Log Information
                # ------------------

                batches_done = epoch * len(train_loader) + i
                batches_left = opt.epochs * len(train_loader) - batches_done
                time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
                prev_time = time.time()

                print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, aux: %f] [G loss: %f, aux: %f, cycle: %f] ETA: %s" %
                      (epoch, opt.epochs, i, len(train_loader), D_loss.item(), d_cls_loss.item(), G_loss.item(), g_cls_loss.item(), g_rec_loss, time_left))

                if batches_done % opt.sample_interval == 0:
                    save_sample(opt.dataset, val_loader, batches_done, generator, FloatTensor)

                if batches_done % opt.checkpoint_interval == 0:
                    torch.save(Generator.state_dict(), "checkpoints/%s/G_%d.pth" % (opt.dataset, epoch))

    torch.save(Generator.state_dict(), "checkpoints/%s/shared_E_done.pth" % opt.dataset)
    print("Training Process has been Done!")


if __name__ == '__main__':
    train()
