import numpy as np

import torch

from torch.autograd import Variable
from torchvision.utils import save_image

from config import parse_args
from model import Generator


def infer():
    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    opt = parse_args()
    generator = Generator(opt)
    generator.load_state_dict(torch.load(opt.load_model))

    if cuda:
        generator.cuda()

    z = Variable(Tensor(np.random.normal(0, 1, (opt.batch_size, opt.latent_dim))))
    gen_imgs = generator(z)

    save_image(gen_imgs.data[:25], "images/infer.png", nrow=5, normalize=True)


if __name__ == '__main__':
    infer()
