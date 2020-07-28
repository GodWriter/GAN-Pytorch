import torch

from PIL import Image
from torch.autograd import Variable
from torchvision.utils import save_image
import torchvision.transforms as transforms

from config import parse_args
from model import Generator, Discriminator
from torchsummary import summary


def load_img(opt):
    # pre-process the test image
    transform = transforms.Compose([transforms.Resize((opt.img_height, opt.img_width), Image.BICUBIC),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    img = Image.open(opt.test_img)
    img = transform(img)

    return img


def display_network(opt):
    cuda = True if torch.cuda.is_available() else False

    generator = Generator(opt)
    # generator.load_state_dict(torch.load(opt.load_model))
    discriminator = Discriminator(opt)

    if cuda:
        generator.cuda()
        discriminator.cuda()

    # print(*discriminator.output_shape)
    summary(generator, (opt.channels, opt.img_height, opt.img_width))
    summary(discriminator, (opt.channels, opt.img_height, opt.img_width))


def infer(opt):
    cuda = True if torch.cuda.is_available() else False
    FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    G_AB = Generator(opt)
    G_BA = Generator(opt)

    G_AB.load_state_dict(torch.load(opt.load_model.split(',')[0]))
    G_BA.load_state_dict(torch.load(opt.load_model.split(',')[1]))

    if cuda:
        G_AB.cuda()
        G_BA.cuda()

    sample = load_img(opt)
    sample = Variable(sample.unsqueeze(0).type(FloatTensor))
    gen_img_B = G_AB(sample)
    gen_img_A = G_BA(sample)

    sample = torch.cat((sample.data, gen_img_B.data, gen_img_A.data), -1)
    save_image(sample, "images/infer.png", nrow=1, normalize=True)


if __name__ == '__main__':
    opt = parse_args()
    # infer(opt)
    display_network(opt)
