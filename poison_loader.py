'''Train CIFAR10 with PyTorch.'''
from sklearn import datasets
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
import numpy as np
from PIL import Image
from copy import deepcopy
import torch


class folder_load(Dataset):
    '''
    poison_rate: the proportion of poisoned images in training set, controlled by seed.
    non_poison_indices: indices of images that are clean.
    '''
    def __init__(self, path,  T, poison_rate=1, seed=0, non_poison_indices=None):
        self.T =  T
        self.targets = datasets.CIFAR10(root='~/data/', train=True).targets
        self.trainls = [str(i) for i in range(50000)]
        self.path = path
        self.PILimgs = []
        self.classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
        for item in self.trainls:
            img = Image.open(self.path + item + '.png')
            im_temp = deepcopy(img)
            self.PILimgs.append(im_temp)
            img.close()

        self.c10  = datasets.CIFAR10('../data/', train=True)
        self.PILc10 = [item[0] for item in self.c10]
        if non_poison_indices is not None:
            self.non_poison_indices = non_poison_indices
        else:
            np.random.seed(seed)
            self.non_poison_indices = np.random.choice(range(50000), int((1 - poison_rate)*50000), replace=False)
        for idx in self.non_poison_indices:
            self.PILimgs[idx] = self.PILc10[idx]


    def __getitem__(self, index):
        train = self.T(self.PILimgs[index])
        target = self.targets[index]
        return train, target

    def __len__(self):
        return len(self.targets)


class CIFAR10dirty(Dataset):

    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

    def __init__(self, root, poison_rate, seed=0, transform=None, non_poison_indices=None):
        self.transform = transform
        self.c10  = datasets.CIFAR10(root, train=True)
        self.targets = self.c10.targets

        if non_poison_indices is not None:
            self.non_poison_indices = non_poison_indices
        else:
            np.random.seed(seed)
            self.non_poison_indices = np.random.choice(range(50000), int((1 - poison_rate)*50000), replace=False)


    def __getitem__(self, index):
        if index in self.non_poison_indices:
            target = self.targets[index]
            img = self.c10[index][0]
        else:
            target = (self.targets[index]+1)%10
            img = self.c10[index][0]
        if self.transform is not None:
            img = self.transform(img)
        return img, target
    
    def __len__(self):
        return len(self.targets)


class CIFAR10grad(Dataset):

    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

    def __init__(self, root, poison_rate, seed=0, transform=None, non_poison_indices=None):
        self.transform = transform
        self.c10 = datasets.CIFAR10(root, train=True)
        self.targets = self.c10.targets

        if non_poison_indices is not None:
            self.non_poison_indices = non_poison_indices
        else:
            np.random.seed(seed)
            self.non_poison_indices = np.random.choice(range(50000), int((1 - poison_rate)*50000), replace=False)

        self.poison_data = torch.load('../data/CIFAR10Poison/Linf.grad.24.0')[0]
        self.totensor = transforms.Compose([transforms.ToTensor()])
        self.toimg = transforms.Compose([transforms.ToPILImage()])

    def __getitem__(self, index):
        if index in self.non_poison_indices:
            target = self.targets[index]
            img = self.c10[index][0]
        else:
            target = self.targets[index]
            img = self.poison_data[index]
            img = self.toimg(img)

        if self.transform is not None:
            img = self.transform(img)
        return img, target
    
    def __len__(self):
        return len(self.targets)

