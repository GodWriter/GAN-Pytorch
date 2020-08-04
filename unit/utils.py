import os
import glob
import torch
import random
import imageio
import numpy as np

from PIL import Image
from torch.autograd import Variable
from torchvision.utils import save_image, make_grid


class ReplayBuffer:
    def __init__(self, max_size=50):
        assert max_size > 0, "Empty buffer. Be careful."
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []

        # element.size = batch_size
        # if buffer is full, replace data in buffer randomly
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)

        return Variable(torch.cat(to_return))


def stack_img(image_path):
    imgs = []

    files = sorted(glob.glob("%s/*.*" % image_path))
    for file in files:
        imgs.append(np.array(Image.open(file)))

    result_img = np.vstack(tuple(imgs))
    Image.fromarray(result_img).save(os.path.join(image_path, 'result.png'))


def create_gif(image_path):
    frames = []
    gif_name = os.path.join("images", 'display.gif')

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
        img = img.resize((img.size[0] // 2, img.size[1] // 2))
        img.save(img_path)


def save_sample(dataset, test_loader, batches_done, E1, E2, G1, G2, FloatTensor):
    X1, X2 = next(iter(test_loader))

    X1 = Variable(X1.type(FloatTensor))
    X2 = Variable(X2.type(FloatTensor))

    _, Z1 = E1(X1)
    _, Z2 = E2(X2)
    fake_X1 = G1(Z2)
    fake_X2 = G2(Z1)

    samples = torch.cat((X1.data, fake_X2.data, X2.data, fake_X1.data), 0)
    save_image(samples, "images/%s/%d.png" % (dataset, batches_done), nrow=5, normalize=True)


if __name__ == "__main__":
    # image_path = "images/example4"
    # resize_img(image_path)
    # create_gif(image_path)

    resize_img("images1/1")
    # stack_img("images/test")
