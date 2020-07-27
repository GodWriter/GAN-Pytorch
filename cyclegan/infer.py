import torch

from PIL import Image
from torch.autograd import Variable
from torchvision.utils import save_image
import torchvision.transforms as transforms

from config import parse_args
from model import Generator, Discriminator
from torchsummary import summary


def apply_center_mask(opt, img):
    top_left = (opt.img_size - opt.mask_size) // 2

    masked_img = img.clone()
    masked_img[:, top_left: top_left + opt.mask_size, top_left: top_left + opt.mask_size] = 1

    return masked_img, top_left


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

    generator = Generator(opt.channels, opt.channels)
    # generator.load_state_dict(torch.load(opt.load_model))
    discriminator = Discriminator(opt.channels)

    if cuda:
        generator.cuda()
        discriminator.cuda()

    summary(generator, (opt.channels, opt.img_height, opt.img_width))
    summary(discriminator, [(opt.channels, opt.img_height, opt.img_width), (opt.channels, opt.img_height, opt.img_width)])


def infer(opt):
    cuda = True if torch.cuda.is_available() else False
    FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    generator = Generator()
    generator.load_state_dict(torch.load(opt.load_model))

    if cuda:
        generator.cuda()

    sample = load_img(opt)
    sample = Variable(sample.unsqueeze(0).type(FloatTensor))
    gen_img = generator(sample)

    sample = torch.cat((sample.data, gen_img.data), -1)
    save_image(sample, "images/infer.png", nrow=1, normalize=True)


if __name__ == '__main__':
    opt = parse_args()
    infer(opt)
    # display_network(opt)
