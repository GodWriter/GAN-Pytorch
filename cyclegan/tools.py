import os
import cv2
import torch
import imageio
import argparse
import numpy as np

from PIL import Image
from matplotlib import pyplot as plt

from torch.autograd import Variable
from torchvision.utils import save_image
import torchvision.transforms as transforms
from model import Generator


def load_img(sample):
    # pre-process the test image
    transform = transforms.Compose([transforms.Resize((opt.img_height*2, opt.img_width)),
                                    transforms.ToTensor(), 
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    img = Image.open(sample)
    img = transform(img)

    return img

def transfer(opt):
    cuda = True if torch.cuda.is_available() else False
    FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    G_AB = Generator(opt)
    G_AB.load_state_dict(torch.load(opt.model_dir))

    if cuda:
        G_AB.cuda()
    
    samples = os.listdir(opt.img_save_path)
    cnt = 0

    for i in range(len(samples)):
        sample = load_img(os.path.join(opt.img_save_path, str(cnt) + '.jpg'))
        sample = Variable(sample.unsqueeze(0).type(FloatTensor))

        gen_img_B = G_AB(sample)
        sample = torch.cat((sample.data, gen_img_B.data), -1)

        save_path = os.path.join(opt.img_transfer_path, str(cnt) + '.jpg')
        save_image(sample, save_path, nrow=1, normalize=True)

        cnt += 1

def generate_img(opt):
    # read video
    vc = cv2.VideoCapture(opt.video_path)
    if vc.isOpened():
        isOpen = True
        print("read success!")
    else:
        isOpen = False
        print("read failure!")
        return

    num = 0
    while isOpen:
        ret, frame = vc.read()

        if frame is None:
            break
        if ret:
            cv2.imwrite(os.path.join(opt.img_save_path, str(num) + '.jpg'), frame)
            num += 1
    vc.release()

def create_gif(opt):
    frames = []
    gif_name = os.path.join(opt.path, 'display.gif')
    image_list = os.listdir(opt.img_transfer_path)

    cnt = 0
    for i in range(len(image_list) // opt.frame):
        frames.append(imageio.imread(os.path.join(opt.img_transfer_path, str(cnt) + '.jpg')))
        cnt += opt.frame
    imageio.mimsave(gif_name, frames, 'GIF', duration=opt.duration)

def generate_gif(opt):
    os.makedirs(opt.img_save_path, exist_ok=True)
    os.makedirs(opt.img_transfer_path, exist_ok=True)

    generate_img(opt)
    transfer(opt)
    create_gif(opt)


if __name__ == "__main__":
    # config
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', type=str, default='video/test.mp4', help='path of video')
    parser.add_argument('--path', type=str, default='video', help='path of video')
    parser.add_argument('--img_save_path', type=str, default='video/source', help='path to save the original images')
    parser.add_argument('--img_transfer_path', type=str, default='video/transfer', help='path to save the transferred images')
    parser.add_argument('--img_height', type=int, default=64, help='size of image height')
    parser.add_argument('--img_width', type=int, default=64, help='size of image width')
    parser.add_argument('--channels', type=int, default=3, help='number of image channels')
    parser.add_argument('--num_residual_blocks', type=int, default=7, help='number of residual blocks in generator')
    parser.add_argument('--model_dir', type=str, default='checkpoints/G_AB_91.pth', help='model to load')
    parser.add_argument('--frame', type=int, default=3, help='gif speed, larger is faster')
    parser.add_argument('--duration', type=float, default=0.2, help='gif speed, larger is faster')
    opt = parser.parse_args()

    generate_gif(opt)
