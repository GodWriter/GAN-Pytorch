import os
import time
import torch
import itertools
import datetime

from torch.autograd import Variable

from config import parse_args
from utils import ReplayBuffer, save_sample
from model import Generator, Discriminator, Encoder, ResidualBlock, LambdaLR, weights_init_normal
from dataloader import commic2human_loader


def train():
    os.makedirs("images", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    opt = parse_args()
    cuda = True if torch.cuda.is_available() else False
    FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    # get dataloader
    train_loader = commic2human_loader(opt, mode='train')
    test_loader = commic2human_loader(opt, mode='test')

    # Dimensionality
    input_shape = (opt.channels, opt.img_height, opt.img_width)
    shared_dim = opt.dim * (2 ** opt.n_downsample)

    # Initialize generator and discriminator
    shared_E = ResidualBlock(in_channels=shared_dim)
    E1 = Encoder(dim=opt.dim, n_downsample=opt.n_downsample, shared_block=shared_E)
    E2 = Encoder(dim=opt.dim, n_downsample=opt.n_downsample, shared_block=shared_E)

    shared_G = ResidualBlock(in_channels=shared_dim)
    G1 = Generator(dim=opt.dim, n_upsample=opt.n_downsample, shared_block=shared_G)
    G2 = Generator(dim=opt.dim, n_upsample=opt.n_downsample, shared_block=shared_G)

    D1 = Discriminator(input_shape)
    D2 = Discriminator(input_shape)

    # Initialize weights
    E1.apply(weights_init_normal)
    E2.apply(weights_init_normal)
    G1.apply(weights_init_normal)
    G2.apply(weights_init_normal)
    D1.apply(weights_init_normal)
    D2.apply(weights_init_normal)

    # Loss function
    adversarial_loss = torch.nn.MSELoss()
    pixel_loss = torch.nn.L1Loss()

    if cuda:
        E1 = E1.cuda()
        E2 = E2.cuda()
        G1 = G1.cuda()
        G2 = G2.cuda()
        D1 = D1.cuda()
        D2 = D2.cuda()
        adversarial_loss = adversarial_loss.cuda()
        pixel_loss = pixel_loss.cuda()

    # Optimizers
    optimizer_G = torch.optim.Adam(itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    # Learning rate update schedulers
    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(opt.epochs, 0, opt.decay_epoch).step)
    lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=LambdaLR(opt.epochs, 0, opt.decay_epoch).step)
    lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=LambdaLR(opt.epochs, 0, opt.decay_epoch).step)

    # Buffers of previously generated samples
    gen_A_buffer = ReplayBuffer()
    gen_B_buffer = ReplayBuffer()

    prev_time = time.time()
    for epoch in range(opt.epochs):
        for i, (img_A, img_B) in enumerate(train_loader):

            # Model inputs
            img_A = Variable(img_A.type(FloatTensor))
            img_B = Variable(img_B.type(FloatTensor))

            # Adversarial ground truths
            valid = Variable(FloatTensor(img_A.shape[0], *D_A.output_shape).fill_(1.0), requires_grad=False)
            fake = Variable(FloatTensor(img_A.shape[0], *D_A.output_shape).fill_(0.0), requires_grad=False)

            # Configure input
            gen_A = G_BA(img_B)
            gen_B = G_AB(img_A)

            recov_A = G_BA(gen_B)
            recov_B = G_AB(gen_A)

            gen_A_ = gen_A_buffer.push_and_pop(gen_A)
            gen_B_ = gen_B_buffer.push_and_pop(gen_B)

            # ------------------
            # Train Generator
            # ------------------

            G_AB.train()
            G_BA.train()

            optimizer_G.zero_grad()

            # Identity loss
            id_loss_A = identity_loss(G_BA(img_A), img_A)
            id_loss_B = identity_loss(G_AB(img_B), img_B)
            id_loss = (id_loss_A + id_loss_B) / 2

            # Adversarial loss
            g_adv_AB = adversarial_loss(D_B(gen_B), valid)
            g_adv_BA = adversarial_loss(D_A(gen_A), valid)
            g_adv = (g_adv_AB + g_adv_BA)/ 2

            # Cycle loss
            cyc_loss_A = cycle_loss(img_A, recov_A)
            cyc_loss_B = cycle_loss(img_B, recov_B)
            cyc_loss = (cyc_loss_A + cyc_loss_B) / 2

            # generator loss
            g_loss = g_adv + opt.lambda_cyc * cyc_loss + opt.lambda_id * id_loss
            g_loss.backward()
            optimizer_G.step()

            # ----------------------
            # Train Discriminator A
            # ----------------------

            optimizer_D_A.zero_grad()

            real_loss = adversarial_loss(D_A(img_A), valid)
            fake_loss = adversarial_loss(D_A(gen_A_.detach()), fake)
            d_loss_A = (real_loss + fake_loss) / 2

            d_loss_A.backward()
            optimizer_D_A.step()

            # ----------------------
            # Train Discriminator B
            # ----------------------

            optimizer_D_B.zero_grad()

            real_loss = adversarial_loss(D_B(img_B), valid)
            fake_loss = adversarial_loss(D_B(gen_B_.detach()), fake)
            d_loss_B = (real_loss + fake_loss) / 2

            d_loss_B.backward()
            optimizer_D_B.step()

            d_loss = (d_loss_A + d_loss_B) / 2

            # ------------------
            # Log Information
            # ------------------

            batches_done = epoch * len(train_loader) + i
            batches_left = opt.epochs * len(train_loader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()

            print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, adv: %f, cycle: %f, identity: %f] ETA: %s" %
                  (epoch, opt.epochs, i, len(train_loader), d_loss.item(), g_loss.item(), g_adv.item(), cyc_loss.item(), id_loss.item(), time_left))

            if batches_done % opt.sample_interval == 0:
                save_sample(test_loader, batches_done, G_AB, G_BA, FloatTensor)

            if batches_done % opt.checkpoint_interval == 0:
                torch.save(G_AB.state_dict(), "checkpoints/G_AB_%d.pth" % epoch)
                torch.save(G_BA.state_dict(), "checkpoints/G_BA_%d.pth" % epoch)
                # torch.save(discriminator.state_dict(), "checkpoints/discriminator_%d.pth" % epoch)

        # Update learning rates
        lr_scheduler_G.step()
        lr_scheduler_D_A.step()
        lr_scheduler_D_B.step()

    torch.save(G_AB.state_dict(), "checkpoints/G_AB_done.pth")
    torch.save(G_BA.state_dict(), "checkpoints/G_BA_done.pth")
    print("Training Process has been Done!")


if __name__ == '__main__':
    train()
