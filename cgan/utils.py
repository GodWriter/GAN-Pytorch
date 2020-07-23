import os
import imageio
import numpy as np

from torch.autograd import Variable
from torchvision.utils import save_image


def create_gif(image_path):
    frames = []
    gif_name = os.path.join("images", 'mnist2.gif')

    image_list = os.listdir(image_path)
    sorted(image_list)

    for image_name in image_list:
        frames.append(imageio.imread(os.path.join(image_path, image_name)))

    imageio.mimsave(gif_name, frames, 'GIF', duration=0.1)


def sample_image(opt, n_row, batches_done, generator, FloatTensor, LongTensor):
    """Save a grid of generated digits ranging from 0 to n_classes"""
    z = Variable(FloatTensor(np.random.normal(0, 1, (n_row ** 2, opt.latent_dim))))
    labels = np.array([num for _ in range(n_row) for num in range(n_row)])
    labels = Variable(LongTensor(labels))

    gen_imgs = generator(z, labels)
    save_image(gen_imgs.data,  "images/%d.png" % batches_done, nrow=n_row, normalize=True)




if __name__ == "__main__":
    image_path = "images/example2"
    create_gif(image_path)
