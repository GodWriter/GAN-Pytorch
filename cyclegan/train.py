import os
import time
import torch
import itertools
import datetime

from torch.autograd import Variable

from config import parse_args
from utils import ReplayBuffer, LambdaLR, save_sample
from model import Generator, Discriminator, weights_init_normal
from dataloader import monet2photo_loader


def train():
    os.makedirs("images", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    opt = parse_args()
    cuda = True if torch.cuda.is_available() else False
    FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    # get dataloader
    train_loader = monet2photo_loader(opt, mode='train')
    val_loader = monet2photo_loader(opt, mode='test')

    # Initialize generator and discriminator
    G_AB = Generator(opt)
    G_BA = Generator(opt)
    D_A = Discriminator(opt)
    D_B = Discriminator(opt)

    # Initialize weights
    G_AB.apply(weights_init_normal)
    G_BA.apply(weights_init_normal)
    D_A.apply(weights_init_normal)
    D_B.apply(weights_init_normal)

    # Loss function
    adversarial_loss = torch.nn.MSELoss()
    cycle_loss = torch.nn.L1Loss()
    identity_loss = torch.nn.L1Loss()

    if cuda:
        G_AB.cuda()
        G_BA.cuda()
        D_A.cuda()
        D_B.cuda()
        adversarial_loss.cuda()
        cycle_loss.cuda()
        identity_loss.cuda()

    # Optimizers
    optimizer_G = torch.optim.Adam(itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    # Learning rate update schedulers
    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(opt.epochs, 0, opt.decay_epoch).step)
    lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=LambdaLR(opt.epochs, 0, opt.decay_epoch).step)
    lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=LambdaLR(opt.epochs, 0, opt.decay_epoch).step)

    # Buffers of previously generated samples
    fake_A_buffer = ReplayBuffer()
    fake_B_buffer = ReplayBuffer()

    prev_time = time.time()
    for epoch in range(opt.epochs):
        for i, (img_A, img_B) in enumerate(train_loader):

            # Model inputs
            img_A = Variable(img_A.type(FloatTensor))
            img_B = Variable(img_B.type(FloatTensor))

            # Adversarial ground truths
            valid = Variable(FloatTensor(img_A.shape[0], *patch).fill_(1.0), requires_grad=False)
            fake = Variable(FloatTensor(img_A.shape[0], *patch).fill_(0.0), requires_grad=False)

            # Configure input
            gen_imgs = generator(img_A)

            # ------------------
            # Train Discriminator
            # ------------------

            optimizer_D.zero_grad()

            real_loss = adversarial_loss(discriminator(img_B, img_A), valid)
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach(), img_A), fake)
            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()

            # ------------------
            # Train Generator
            # ------------------

            if i % opt.n_critic == 0:
                optimizer_G.zero_grad()

                # Loss for generator
                g_adv = adversarial_loss(discriminator(gen_imgs, img_A), valid)
                g_pixel = pixelwise_loss(gen_imgs, img_B)

                g_loss = g_adv + opt.lambda_pixel * g_pixel

                # Update parameters
                g_loss.backward()
                optimizer_G.step()

            # ------------------
            # Log Information
            # ------------------

            batches_done = epoch * len(train_loader) + i
            batches_left = opt.epochs * len(train_loader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()

            print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G adv: %f, pixel: %f] ETA: %s" %
                  (epoch, opt.epochs, i, len(train_loader), d_loss.item(), g_adv.item(), g_pixel.item(), time_left))

            if batches_done % opt.sample_interval == 0:
                save_sample(val_loader, batches_done, generator, FloatTensor)

            if batches_done % opt.checkpoint_interval == 0:
                torch.save(generator.state_dict(), "checkpoints/generator_%d.pth" % epoch)
                # torch.save(discriminator.state_dict(), "checkpoints/discriminator_%d.pth" % epoch)

    torch.save(generator.state_dict(), "checkpoints/generator_done.pth")
    print("Training Process has been Done!")


if __name__ == '__main__':
    train()
