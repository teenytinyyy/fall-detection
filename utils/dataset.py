from typing import List, Optional
import random
import csv
import cv2
import numpy as np
import torch
import torchvision

from torchvision.io import read_image
from torch.utils.data import Dataset
from utils import image as img_utils


class ImageDataset(Dataset):
    def __init__(self, label_path, transform=None):

        self.labels = []
        self.imgs = []

        self.transform = transform

        with open(label_path, "r") as r_file:
            rows = csv.reader(r_file)
            for row in rows:
                label = int(row[1])
                self.labels.append(label)

                img = read_image(row[0]).float()

                if self.transform:
                    img = self.transform(img)

                self.imgs.append(img)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.imgs[idx]
        label = self.labels[idx]
        return img, label


def random_indices(start: int, stop: int, step: int = 1) -> List[int]:
    indices = [idx for idx in range(start, stop, step)]

    random.shuffle(indices)

    return indices


def uniform_indices(start: int, stop: int, step: int = 1) -> List[int]:
    indices = [idx for idx in range(start, stop, step)]

    return indices


def read_dataset(input_path: str, img_size: int = 227, test_ratio: float = 0.2, balance_rate: int = 8, balance_cls: List[int] = [], transform: Optional[torchvision.transforms.RandomAffine] = None):
    data = []
    targets = []
    file_list = []

    info = {}

    with open(input_path, "r") as r_file:
        rows = csv.reader(r_file)
        for row in rows:
            img = cv2.imread(row[0])
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # img = np.expand_dims(img, axis=-1)
            img = img_utils.resize(img, img_size)
            # img = np.expand_dims(img, axis=-1)
            img = np.swapaxes(img, 0, 2)
            img = torch.from_numpy(img)

            label = int(row[1])
            target = [0] * 2

            if label not in info:
                info[label] = 0

            info[label] += 1

            target[label] = 1

            data.append(img.numpy())
            targets.append(target)
            file_list.append(row[0])

            if balance_rate > 1 and transform is not None and label in balance_cls:
                for _ in range(balance_rate):
                    data.append(transform(img).numpy())
                    targets.append(target)
                    file_list.append(row[0])
                    info[label] += 1

    data = np.array(data)

    data = torch.from_numpy(data).float()

    targets = np.array(targets)
    targets = torch.from_numpy(targets).long()

    file_list = np.array(file_list)

    indices = random_indices(0, len(data))

    train_data = data[indices[:int(len(indices) * (1 - test_ratio))]]
    train_targets = targets[indices[:int(len(indices) * (1 - test_ratio))]]
    train_file_list = file_list[indices[:int(len(indices) * (1 - test_ratio))]]

    test_data = data[indices[-int(len(indices) * (test_ratio)):]]
    test_targets = targets[indices[-int(len(indices) * (test_ratio)):]]
    test_file_list = file_list[indices[-int(len(indices) * (test_ratio)):]]

    for k, v in info.items():
        print("Class {} has {} data".format(k, v))

    return train_data, train_targets, test_data, test_targets, train_file_list, test_file_list


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def read_cifar_dataset(input_path: str, img_size: int = 227, test_ratio: float = 0.2):

    dataset = unpickle(input_path)

    data = dataset[b'data']
    labels = dataset[b'labels']

    data = np.reshape(data, (-1, 3, 32, 32))
    labels = np.array(labels)

    data = torch.from_numpy(data).float()
    labels = torch.from_numpy(labels).long()

    indices = random_indices(0, len(data))

    train_data = data[indices[:int(len(indices) * (1 - test_ratio))]]
    train_targets = labels[indices[:int(len(indices) * (1 - test_ratio))]]

    test_data = data[indices[-int(len(indices) * (test_ratio)):]]
    test_targets = labels[indices[-int(len(indices) * (test_ratio)):]]

    return train_data, train_targets, test_data, test_targets
