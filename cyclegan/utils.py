import os
import torch
import random
import imageio

from PIL import Image
from torch.autograd import Variable
from torchvision.utils import save_image


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
        img = img.resize((344, 177))
        img.save(img_path)


def save_sample(val_loader, batches_done, generator, FloatTensor):
    img_A, img_B = next(iter(val_loader))

    img_A = Variable(img_A.type(FloatTensor))
    img_B = Variable(img_B.type(FloatTensor))

    gen_imgs = generator(img_A)
    samples = torch.cat((img_A.data, gen_imgs.data, img_B.data), -2)

    save_image(samples, "images/%d.png" % batches_done, nrow=5, normalize=True)


if __name__ == "__main__":
    image_path = "images/example"
    resize_img(image_path)
    create_gif(image_path)

    # resize_img("images/1")
