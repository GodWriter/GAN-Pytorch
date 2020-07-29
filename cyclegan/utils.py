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


class LambdaLR:
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert (n_epochs - decay_start_epoch) > 0, "Decay must start before the training session ends"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)


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
        img = img.resize((342, 342))
        img.save(img_path)


def save_sample(test_loader, batches_done, G_AB, G_BA, FloatTensor):
    img_A, img_B = next(iter(test_loader))
    G_AB.eval()
    G_BA.eval()

    img_A = Variable(img_A.type(FloatTensor))
    img_B = Variable(img_B.type(FloatTensor))
    gen_A = G_BA(img_B)
    gen_B = G_AB(img_A)

    # Arange images along x-axis
    img_A = make_grid(img_A, nrow=5, normalize=True)
    img_B = make_grid(img_B, nrow=5, normalize=True)
    gen_A = make_grid(gen_A, nrow=5, normalize=True)
    gen_B = make_grid(gen_B, nrow=5, normalize=True)

    samples = torch.cat((img_A.data, gen_B.data, img_B.data, gen_A.data), 1)
    save_image(samples, "images/%d.png" % batches_done, normalize=True)


if __name__ == "__main__":
    image_path = "images/example4"
    resize_img(image_path)
    create_gif(image_path)

    # resize_img("images/test")
    # stack_img("images/test")
