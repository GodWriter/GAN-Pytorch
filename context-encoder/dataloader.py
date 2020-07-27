import glob

import numpy as np
import torchvision.transforms as transforms

from PIL import Image
from torch.utils.data import Dataset, DataLoader


class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, img_size=128, mask_size=64, mode='train'):
        self.transform = transforms_
        self.img_size = img_size
        self.mask_size = mask_size
        self.mode = mode
        self.files = sorted(glob.glob("%s/*.jpg" % root))
        self.files = self.files[:-4000] if mode == "train" else self.files[-4000:]
        self.length = len(self.files)

    def apply_random_mask(self, img):
        y1, x1 = np.random.randint(0, self.img_size - self.mask_size, 2)
        y2, x2 = y1 + self.mask_size, x1 + self.mask_size

        masked_part = img[:, y1:y2, x1:x2]
        masked_img = img.clone()
        masked_img[:, y1:y2, x1:x2] = 1

        return masked_img, masked_part

    def apply_center_mask(self, img):
        top_left = (self.img_size - self.mask_size) // 2

        masked_img = img.clone()
        masked_img[:, top_left: top_left + self.mask_size, top_left: top_left + self.mask_size] = 1

        return masked_img, top_left

    def __getitem__(self, index):
        img = Image.open(self.files[index % self.length]).convert("RGB")
        img = self.transform(img)

        # Random mask for training data and center mask for test data
        if self.mode == "train":
            masked_img, aux = self.apply_random_mask(img)
        else:
            masked_img, aux = self.apply_center_mask(img)

        return img, masked_img, aux

    def __len__(self):
        return self.length


def celeba_loader(opt, mode):
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
