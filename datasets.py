"""Super-classes of common datasets to extract id information per image."""
import torch
import torchvision

import torchvision.transforms as transforms
from PIL import Image
import os
import glob

from torchvision.datasets.imagenet import load_meta_file
from torchvision.datasets.utils import verify_str_arg


class CIFAR10(torchvision.datasets.CIFAR10):
    """Super-class CIFAR10 to return image ids with images."""

    def __getitem__(self, index):
        """Getitem from https://pytorch.org/vision/stable/_modules/torchvision/datasets/cifar.html#CIFAR10.

        Args:
            index (int): Index

        Returns:
            tuple: (image, target, idx) where target is index of the target class.

        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def get_target(self, index):
        """Return only the target and its id.

        Args:
            index (int): Index

        Returns:
            tuple: (target, idx) where target is class_index of the target class.

        """
        target = self.targets[index]

        if self.target_transform is not None:
            target = self.target_transform(target)

        return target, index