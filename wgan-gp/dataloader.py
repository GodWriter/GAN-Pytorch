import os
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torchvision import datasets


def mnist_loader(opt):
    # create data directory
    data_path = 'data/mnist'
    os.makedirs(data_path, exist_ok=True)

    # pre-process the data
    transform = transforms.Compose([transforms.Resize(opt.img_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.5], [0.5])])

    mnist_train = datasets.MNIST(data_path,
                                 train=True,
                                 download=True,
                                 transform=transform)

    # create the data_loader
    data_loader = DataLoader(mnist_train,
                             batch_size=opt.batch_size,
                             shuffle=True)

    return data_loader
