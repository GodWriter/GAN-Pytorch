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


def save_sample(val_loader, batches_done, generator, FloatTensor):
    img_A, img_B = next(iter(val_loader))

    img_A = Variable(img_A.type(FloatTensor))
    img_B = Variable(img_B.type(FloatTensor))

    gen_imgs = generator(img_A)
    samples = torch.cat((img_A.data, gen_imgs.data, img_B.data), -2)

    save_image(samples, "images/%d.png" % batches_done, nrow=5, normalize=True)


if __name__ == "__main__":
    image_path = "images/example1"
    resize_img(image_path)
    create_gif(image_path)
