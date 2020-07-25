import os
import glob

import numpy as np
import torchvision.transforms as transforms

from PIL import Image
from torch.utils.data import Dataset, DataLoader


class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, mode='train'):
        self.transform = transforms_
        self.mode = mode

        self.files = sorted(glob.glob(os.path.join(root, mode) + '/*.*'))
        if mode == 'train':
            self.files.extend(sorted(glob.glob(os.path.join(root, 'test') + '/*.*')))

        self.length = len(self.files)

    def __getitem__(self, index):
        img = Image.open(self.files[index % self.length])
        w, h = img.size

        img_A = img.crop((0, 0, w // 2, h))
        img_B = img.crop((w // 2, 0, w, h))

        # flip left to right randomly
        if np.random.random() < 0.5:
            img_A = Image.fromarray(np.array(img_A)[:, ::-1, :], "RGB")
            img_B = Image.fromarray(np.array(img_B)[:, ::-1, :], "RGB")

        img_A = self.transform(img_A)
        img_B = self.transform(img_B)

        return img_A, img_B

    def __len__(self):
        return self.length


def facades_loader(opt, mode):
    data_path = 'data/%s' % opt.dataset

    # pre-process the data
    transform = transforms.Compose([transforms.Resize((opt.img_size, opt.img_size), Image.BICUBIC),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    loader = ImageDataset(data_path,
                          transforms_=transform,
                          mode=mode)

    # create the data_loader
    if mode == 'train':
        data_loader = DataLoader(loader,
                                 batch_size=opt.batch_size,
                                 shuffle=True)
    elif mode == 'test':
        data_loader = DataLoader(loader,
                                 batch_size=12,
                                 shuffle=True)

    return data_loader
