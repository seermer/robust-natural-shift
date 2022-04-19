from torchvision import datasets
from torch.utils import data
from PIL import Image
import numpy as np
import os

BASE_PATH = 'labeled_data'


class NumpyDataset(data.Dataset):
    def __init__(self, path, transform=None):
        if not ((isinstance(path, tuple) or isinstance(path, list)) and len(path) in (1, 2)):
            raise ValueError('NumpyDataset requires path to be a list/tuple of paths to two .npy or there')
        for p in path:
            if not (os.path.exists(p) and os.path.isfile(p) and p.lower().endswith('.npy')):
                raise ValueError('path must points to .npy for NumpyDataset')
        self.np_x = np.load(path[0])
        self.np_y = np.load(path[1]) if len(path) == 2 else None
        self.transforms = transform

    def __getitem__(self, index):
        img = self.np_x[index],

        img = Image.fromarray(img)
        if self.transforms is not None:
            img = self.transforms(img)
        if self.np_y is not None:
            return img, self.np_y[index]
        else:
            return img, 1  # 1 is used to align with other datasets, must be ignored when using

    def __len__(self):
        return len(self.np_y)


def get_dataset(name: str,
                transform=None,
                val_transform=None,
                custom_train_path: list = None,
                custom_val_path: list = None,
                custom_mode: str = None):
    """
    load and return a dataset,
    note that when using custom dataset

    :param name: name of dataset
    :param transform: transform to apply on train dataset
    :param val_transform: transform to apply on val dataset
    :param custom_train_path: path to custom dataset, must not be None when name is custom:
        when custom_mode is 'npy', this path should be a tuple or list of one or two strings,
            when length is two, the first will be interpreted as features, the second will be interpreted as labels;
            when length is one, dataset will contain label of 1 which needs to be ignored when using,
        when custom_mode is 'npy_folder', this path should be path to the directory containing npy files
        when name is not custom, this param is ignored
    :param custom_val_path: path to custom dataset, if None when name is custom, no val set will be returned,
        same format as custom_train_path
    :param custom_mode: one of 'npy', 'npy_folder', must not be None when name is custom
    :return: dataset
    """
    name = name.lower()
    if name == 'custom':
        custom_train_path = [os.path.join('custom_data', p) for p in custom_train_path]
        custom_val_path = [os.path.join('custom_data', p) for p in custom_val_path]
        if (custom_mode is None) or (custom_train_path is None):
            raise ValueError('custom_dir and custom_mode must not be None when name is custom')
        if custom_mode == 'npy':
            if custom_val_path is None:
                return NumpyDataset(custom_train_path, transform), None
            else:
                return NumpyDataset(custom_train_path, transform), NumpyDataset(custom_val_path, val_transform)
        elif custom_mode == 'npy_folder':
            assert len(custom_train_path) == len(custom_val_path) == 1, '"npy_folder" requires only 1 path for each'
            custom_train_path, custom_val_path = custom_train_path[0], custom_val_path[0]
            npy_folder_loader = lambda p: Image.fromarray(np.load(p))
            if custom_val_path is None:
                return (datasets.DatasetFolder(root=custom_train_path, loader=npy_folder_loader,
                                               extensions=('.npy',), transform=transform),
                        None)
            else:
                return (datasets.DatasetFolder(root=custom_train_path, loader=npy_folder_loader,
                                               extensions=('.npy',), transform=transform),
                        datasets.DatasetFolder(root=custom_val_path, loader=npy_folder_loader,
                                               extensions=('.npy',), transform=val_transform))
    elif name == 'cifar10':
        path = os.path.join(os.path.curdir, BASE_PATH, 'cifar10')
        return (datasets.CIFAR10(path, transform=transform, train=True, download=True),
                datasets.CIFAR10(path, transform=val_transform, train=False))
    elif name == 'cifar100':
        path = os.path.join(os.path.curdir, BASE_PATH, 'cifar100')
        return (datasets.CIFAR100(path, transform=transform, train=True, download=True),
                datasets.CIFAR100(path, transform=val_transform, train=False))
    elif name == 'mnist':
        path = os.path.join(os.path.curdir, BASE_PATH, 'mnist')
        return (datasets.MNIST(path, transform=transform, train=True, download=True),
                datasets.CIFAR10(path, transform=val_transform, train=False))
    elif name == 'svhn':
        path = os.path.join(os.path.curdir, BASE_PATH, 'svhn')
        return (datasets.SVHN(path, transform=transform, split='train', download=True),
                datasets.SVHN(path, transform=val_transform, split='val'))
    elif name == 'svhn_extra':
        path = os.path.join(os.path.curdir, BASE_PATH, 'svhn')
        return (datasets.SVHN(path, transform=transform, split='extra', download=True),
                None)
