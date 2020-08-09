import os
import glob
import torch
import random

import numpy as np
import torchvision.transforms as transforms

from PIL import Image
from torch.utils.data import Dataset, DataLoader


class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, mode='train', attribute=None):
        self.transform = transforms_
        self.selected_attrs = attribute

        self.files = sorted(glob.glob("%s/*.jpg" % root))
        self.files = self.files[:-2000] if mode == "train" else self.files[-2000:]
        self.label_path = glob.glob("%s/*.txt" % root)[0]
        self.annotations = self.get_annotations()

        self.length = len(self.files)

    def get_annotations(self):
        annotations = {}

        lines = [line.rstrip() for line in open(self.label_path, "r")]
        self.label_names = lines[1].split()

        for _, line in enumerate(lines[2:]):
            labels = []
            filename, *values = line.split()
            for attr in self.selected_attrs:
                idx = self.label_names.index(attr)
                labels.append(1 * (values[idx] == "1"))
            annotations[filename] = labels

        return annotations

    def __getitem__(self, index):
        file_path = self.files[index % self.length]
        file_name = file_path.split("/")[-1]

        img = self.transform(Image.open(file_path))
        label = self.annotations[file_name]
        label = torch.FloatTensor(np.array(label))

        return img, label

    def __len__(self):
        return self.length


def celeba_loader(opt, mode):
    data_path = 'data/%s' % opt.dataset

    # pre-process the data
    transform = transforms.Compose([transforms.Resize((opt.img_height, opt.img_width), Image.BICUBIC),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    loader = ImageDataset(data_path,
                          transforms_=transform,
                          mode=mode,
                          attribute=opt.selected_attrs)

    # create the data_loader
    if mode == 'train':
        data_loader = DataLoader(loader,
                                 batch_size=opt.batch_size,
                                 shuffle=True)
    elif mode == 'val':
        data_loader = DataLoader(loader,
                                 batch_size=10,
                                 shuffle=True)

    return data_loader
