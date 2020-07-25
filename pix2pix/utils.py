import os
import torch
import imageio

from PIL import Image
from torch.autograd import Variable
from torchvision.utils import save_image


def create_gif(image_path):
    frames = []
    gif_name = os.path.join("images", 'mnist1.gif')

    image_list = os.listdir(image_path)
    sorted(image_list)

    for image_name in image_list:
        frames.append(imageio.imread(os.path.join(image_path, image_name)))

    imageio.mimsave(gif_name, frames, 'GIF', duration=0.1)


def resize_img(path):
    names = os.listdir(path)
    for name in names:
        img_path = os.path.join(path, name)
        img = Image.open(img_path)
        img = img.resize((172, 172))
        img.save(img_path)


def save_sample(opt, test_loader, batches_done, generator, FloatTensor):
    samples, masked_samples, i = next(iter(test_loader))
    samples = Variable(samples.type(FloatTensor))
    masked_samples = Variable(masked_samples.type(FloatTensor))
    i = i[0].item()

    gen_mask = generator(masked_samples)
    filled_samples = masked_samples.clone()
    filled_samples[:, :, i: i+opt.mask_size, i: i+opt.mask_size] = gen_mask

    sample = torch.cat((masked_samples.data, filled_samples.data, samples.data), -2)
    save_image(sample, "images/%d.png" % batches_done, nrow=6, normalize=True)


if __name__ == "__main__":
    image_path = "images/example1"
    resize_img(image_path)
    create_gif(image_path)
