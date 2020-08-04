import os
import glob
import random

import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader


class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode='train'):
        self.transform = transforms_
        self.unaligned = unaligned

        self.files_A = sorted(glob.glob(os.path.join(root, "%sA" % mode) + "/*.*"))
        self.files_B = sorted(glob.glob(os.path.join(root, "%sB" % mode) + "/*.*"))

        self.len_A = len(self.files_A)
        self.len_B = len(self.files_B)

    def __getitem__(self, index):
        img_A = Image.open(self.files_A[index % self.len_A])

        if self.unaligned:
            img_B = Image.open(self.files_B[random.randint(0, self.len_B-1)])
        else:
            img_B = Image.open(self.files_B[index % self.len_B])

        # Convert images to rgb
        if img_A.mode != "RGB":
            img_A = img_A.convert("RGB")
        if img_B.mode != "RGB":
            img_B = img_B.convert("RGB")

        img_A = self.transform(img_A)
        img_B = self.transform(img_B)

        return img_A, img_B

    def __len__(self):
        return min(self.len_A, self.len_B)


def commic2human_loader(opt, mode):
    data_path = 'data/%s' % opt.dataset

    # pre-process the data
    transform = transforms.Compose([transforms.Resize(int(opt.img_height * 1.12), Image.BICUBIC),
                                    transforms.RandomCrop((opt.img_height, opt.img_width)),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    loader = ImageDataset(data_path,
                          transforms_=transform,
                          unaligned=True,
                          mode=mode)

    # create the data_loader
    if mode == 'train':
        data_loader = DataLoader(loader,
                                 batch_size=opt.batch_size,
                                 shuffle=True)
    elif mode == 'test':
        data_loader = DataLoader(loader,
                                 batch_size=5,
                                 shuffle=True)

    return data_loader
