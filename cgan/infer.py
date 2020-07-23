import numpy as np

import torch

from torch.autograd import Variable
from torchvision.utils import save_image

from config import parse_args
from model import Generator


def infer(n_row):
    cuda = True if torch.cuda.is_available() else False
    FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

    opt = parse_args()
    generator = Generator(opt)
    generator.load_state_dict(torch.load(opt.load_model))

    if cuda:
        generator.cuda()

    z = Variable(FloatTensor(np.random.normal(0, 1, (n_row ** 2, opt.latent_dim))))
    labels = np.array([num for _ in range(n_row) for num in range(n_row)])
    labels = Variable(LongTensor(labels))

    gen_imgs = generator(z, labels)
    save_image(gen_imgs.data,  "images/infer.png", nrow=n_row, normalize=True)


if __name__ == '__main__':
    infer(10)
