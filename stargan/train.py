import os
import time
import torch
import datetime

import torch.nn.functional as F
import numpy as np

from torch.autograd import Variable
from config import parse_args
from utils import save_sample
from model import Generator, Discriminator, weights_init_normal
from dataloader import celeba_loader


def criterion_cls(logit, target):
    return F.binary_cross_entropy_with_logits(logit, target, size_average=False) / logit.size(0)


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
    discriminator = Discriminator(opt.channels, c_dim)

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

    label_changes = [((0, 1), (1, 0), (2, 0)),  # Set to black hair
                     ((0, 0), (1, 1), (2, 0)),  # Set to blonde hair
                     ((0, 0), (1, 0), (2, 1)),  # Set to brown hair
                     ((3, -1),),  # Flip gender
                     ((4, -1),),  # Age flip
                    ]

    saved_samples = []
    prev_time = time.time()
    for epoch in range(opt.epochs):
        for i, (imgs, labels) in enumerate(train_loader):

            # Model inputs
            imgs = Variable(imgs.type(FloatTensor))
            labels = Variable(labels.type(FloatTensor))

            # Sample label as generator inputs and Generate fake batch of images
            sampled_c = Variable(FloatTensor(np.random.randint(0, 2, (imgs.size(0), c_dim))))
            fake_imgs = generator(imgs, sampled_c)

            # -----------------------------
            # Train Generators
            # -----------------------------
            pass

            # ----------------------
            # Train Discriminator
            # ----------------------
            pass

            # ------------------
            # Log Information
            # ------------------

            batches_done = epoch * len(train_loader) + i
            batches_left = opt.epochs * len(train_loader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()

            print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] ETA: %s" %
                  (epoch, opt.epochs, i, len(train_loader), (D1_loss + D2_loss).item(), G_loss.item(), time_left))

            if batches_done % opt.sample_interval == 0:
                save_sample(opt.dataset, val_loader, batches_done, E1, E2, G1, G2, FloatTensor)

            if batches_done % opt.checkpoint_interval == 0:
                torch.save(E1.state_dict(), "checkpoints/%s/E1_%d.pth" % (opt.dataset, epoch))
                torch.save(E2.state_dict(), "checkpoints/%s/E2_%d.pth" % (opt.dataset, epoch))
                torch.save(G1.state_dict(), "checkpoints/%s/G1_%d.pth" % (opt.dataset, epoch))
                torch.save(G2.state_dict(), "checkpoints/%s/G2_%d.pth" % (opt.dataset, epoch))

    torch.save(shared_E.state_dict(), "checkpoints/%s/shared_E_done.pth" % opt.dataset)
    torch.save(shared_G.state_dict(), "checkpoints/%s/shared_G_done.pth" % opt.dataset)
    torch.save(E1.state_dict(), "checkpoints/%s/E1_done.pth" % opt.dataset)
    torch.save(E2.state_dict(), "checkpoints/%s/E2_done.pth" % opt.dataset)
    torch.save(G1.state_dict(), "checkpoints/%s/G1_done.pth" % opt.dataset)
    torch.save(G2.state_dict(), "checkpoints/%s/G2_done.pth" % opt.dataset)
    print("Training Process has been Done!")


if __name__ == '__main__':
    train()
