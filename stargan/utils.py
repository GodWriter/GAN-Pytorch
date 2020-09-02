import os
import glob
import torch
import random
import imageio
import numpy as np

from PIL import Image
from torch.autograd import Variable
from torchvision.utils import save_image, make_grid


label_changes = [((0, 1), (1, 0), (2, 0)),  # Set to black hair
                 ((0, 0), (1, 1), (2, 0)),  # Set to blonde hair
                 ((0, 0), (1, 0), (2, 1)),  # Set to brown hair
                 ((3, -1),),  # Flip gender
                 ((4, -1),),  # Age flip
                 ]


def stack_img(image_path):
    imgs = []

    files = sorted(glob.glob("%s/*.*" % image_path))
    for file in files:
        imgs.append(np.array(Image.open(file)))

    result_img = np.vstack(tuple(imgs))
    Image.fromarray(result_img).save(os.path.join(image_path, 'result.png'))


def create_gif(image_path):
    frames = []
    gif_name = os.path.join("images", 'display2.gif')
    image_list = os.listdir(image_path)

    image_id = []
    for name in image_list:
        image_id.append(name[:-4])
    sorted(image_id)

    cnt = 0
    for idx in image_id:
        if cnt % 5 == 0:
            frames.append(imageio.imread(os.path.join(image_path, str(idx) + '.png')))
        cnt += 1

    imageio.mimsave(gif_name, frames, 'GIF', duration=0.1)


def resize_img(path):
    names = os.listdir(path)
    for name in names:
        img_path = os.path.join(path, name)
        img = Image.open(img_path)
        img = img.resize((172, 172))
        img.save(img_path)


def save_sample(dataset, test_loader, batches_done, E1, E2, G1, G2, FloatTensor):
    pass


if __name__ == "__main__":
    image_path = "images/example2"
    # resize_img(image_path)
    create_gif(image_path)
