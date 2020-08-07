import torch

from PIL import Image
from torch.autograd import Variable
from torchvision.utils import save_image
import torchvision.transforms as transforms

from config import parse_args
from model import Encoder, ResidualBlock, Generator, Discriminator
from torchsummary import summary


def load_img(opt):
    # pre-process the test image
    transform = transforms.Compose([transforms.Resize(int(opt.img_height * 1.12), Image.BICUBIC),
                                    transforms.RandomCrop((opt.img_height, opt.img_width)),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    img = Image.open(opt.test_img)
    img = transform(img)

    return img


def display_network(opt):
    cuda = True if torch.cuda.is_available() else False

    # Dimensionality
    input_shape = (opt.channels, opt.img_height, opt.img_width)
    shared_dim = opt.dim * (2 ** opt.n_downsample)

    # Initialize generator and discriminator
    shared_E = ResidualBlock(in_channels=shared_dim)
    E1 = Encoder(dim=opt.dim, n_downsample=opt.n_downsample, shared_block=shared_E)
    E2 = Encoder(dim=opt.dim, n_downsample=opt.n_downsample, shared_block=shared_E)

    shared_G = ResidualBlock(in_channels=shared_dim)
    G1 = Generator(dim=opt.dim, n_upsample=opt.n_upsample, shared_block=shared_G)
    G2 = Generator(dim=opt.dim, n_upsample=opt.n_upsample, shared_block=shared_G)

    D1 = Discriminator(input_shape)
    D2 = Discriminator(input_shape)

    if cuda:
        E1 = E1.cuda()
        E2 = E2.cuda()
        G1 = G1.cuda()
        G2 = G2.cuda()
        D1 = D1.cuda()
        D2 = D2.cuda()

    summary(E1, (opt.channels, opt.img_height, opt.img_width))
    summary(E2, (opt.channels, opt.img_height, opt.img_width))
    summary(G1, (opt.img_height, opt.dim, opt.dim))
    summary(G2, (opt.img_height, opt.dim, opt.dim))
    summary(D1, (opt.channels, opt.img_height, opt.img_width))
    summary(D2, (opt.channels, opt.img_height, opt.img_width))


def infer(opt):
    cuda = True if torch.cuda.is_available() else False
    FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    # Dimensionality
    shared_dim = opt.dim * (2 ** opt.n_downsample)

    # Initialize generator and discriminator
    shared_E = ResidualBlock(in_channels=shared_dim)
    shared_G = ResidualBlock(in_channels=shared_dim)

    E1 = Encoder(dim=opt.dim, n_downsample=opt.n_downsample, shared_block=shared_E)
    G2 = Generator(dim=opt.dim, n_upsample=opt.n_upsample, shared_block=shared_G)

    shared_E.load_state_dict(torch.load(opt.load_model.replace('*', 'shared_E')))
    shared_G.load_state_dict(torch.load(opt.load_model.replace('*', 'shared_G')))
    E1.load_state_dict(torch.load(opt.load_model.replace('*', 'E1')))
    G2.load_state_dict(torch.load(opt.load_model.replace('*', 'G2')))

    if cuda:
        shared_E.cuda()
        shared_G.cuda()
        E1 = E1.cuda()
        G2 = G2.cuda()

    sample = load_img(opt)
    sample = Variable(sample.unsqueeze(0).type(FloatTensor))
    _, Z1 = E1(sample)
    fake_X2 = G2(Z1)

    sample = torch.cat((sample.data, fake_X2.data), -1)
    save_image(sample, "images/infer.png", nrow=1, normalize=True)


if __name__ == '__main__':
    opt = parse_args()
    infer(opt)
    # display_network(opt)
