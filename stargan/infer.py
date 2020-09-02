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
    c_dim = len(opt.selected_attrs)

    generator = Generator(opt.channels, opt.residual_blocks, c_dim)
    discriminator = Discriminator(opt.channels, opt.img_height, c_dim)

    if cuda:
        generator.cuda()
        discriminator.cuda()

    summary(generator, [(opt.channels, opt.img_height, opt.img_width), (c_dim)])
    summary(discriminator, (opt.channels, opt.img_height, opt.img_width))


def infer(opt):
    cuda = True if torch.cuda.is_available() else False
    FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    generator = Generator()
    generator.load_state_dict(torch.load(opt.load_model))

    if cuda:
        generator.cuda()

    sample = load_img(opt)
    sample = Variable(sample.unsqueeze(0).type(FloatTensor))
    label = Variable(opt.label.type(FloatTensor).unsqueeze(0))

    gen_img = generator(sample, label)
    sample = torch.cat((sample.data, gen_img.data), -1)

    save_image(sample, "images/infer.png", nrow=1, normalize=True)


if __name__ == '__main__':
    opt = parse_args()
    infer(opt)
    # display_network(opt)
