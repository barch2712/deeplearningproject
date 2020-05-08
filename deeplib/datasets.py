import math
import random
import time

import numpy as np
import torch
import torch.utils.data
import os
import csv
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.datasets.mnist import MNIST
from torchvision.datasets.cifar import CIFAR10

from const import COVID_LABEL, NORMAL_LABEL, PNEUMONIA_LABEL

BASE_PATH = '~/GLO-4030/datasets/'

get_special_sample = None


def load_mnist(download=False, path=os.path.join(BASE_PATH, 'mnist')):
    train_dataset = MNIST(path, train=True, download=download)
    test_dataset = MNIST(path, train=False, download=download)
    return train_dataset, test_dataset


def load_cifar10(download=False, path=os.path.join(BASE_PATH, 'cifar10')):
    train_dataset = CIFAR10(path, train=True, download=download)
    test_dataset = CIFAR10(path, train=False, download=download)
    return train_dataset, test_dataset


def load_shakespear(path=BASE_PATH, file_name='Shakespeare_data.csv'):
    full_path = os.path.join(path, file_name)
    data = []
    with open(full_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            data.append(row)

    only_lines = []
    for x in data:
        only_lines.append(x[5])
    return only_lines

def load_quotes(path=BASE_PATH, file_name='author-quote.txt'):
    full_path = os.path.join(path, file_name)
    data = []
    file = open(full_path, 'r')
    for line in file:
        author, quote = line.split('\t')
        data.append(quote)
    return data


def train_valid_loaders(dataset, batch_size, train_split=0.8, shuffle=True):
    num_data = len(dataset)
    indices = np.arange(num_data)

    if shuffle:
        random.Random(33).shuffle(indices)

    split = math.floor(train_split * num_data)
    train_idx, valid_idx = indices[:split], indices[split:]

    if get_special_sample == "no_covid":
        a=time.time()
        # if True:
        #     labels = np.asarray([dataset[indice][1] for indice in indices])
        #     with open('shuffled_labels.txt', 'w') as filehandle:
        #         for label in labels:
        #             filehandle.write('%s\n' % label)
        if True:
            labels = []
            with open('shuffled_labels.txt', 'r') as filehandle:
                for line in filehandle:
                    current_place = line[:-1]
                    labels.append(int(current_place))

        labels = np.asarray(labels)
        covid_labels = labels == COVID_LABEL
        train_covid_msk, valid_covid_msk = covid_labels[:split], covid_labels[split:]
        nb_covid_train = sum(train_covid_msk)

        normal_labels = labels == NORMAL_LABEL
        train_normal_msk, valid_normal_msk = normal_labels[:split], normal_labels[split:]
        train_normal_idx, valid_normal_idx = train_idx[train_normal_msk], valid_idx[valid_normal_msk]
        train_normal_nocov_idx = train_normal_idx[:-nb_covid_train]

        pneu_labels = labels == PNEUMONIA_LABEL
        train_pneu_msk, valid_pneu_msk = pneu_labels[:split], pneu_labels[split:]
        train_pneu_idx, valid_pneu_idx = train_idx[train_pneu_msk], valid_idx[valid_pneu_msk]
        train_pneu_nocov_idx = train_pneu_idx[:-nb_covid_train]

        train_sampler = SubsetRandomSampler(np.concatenate((train_normal_nocov_idx, train_pneu_nocov_idx), axis=None))
        valid_sampler = SubsetRandomSampler(np.concatenate((valid_normal_idx, valid_pneu_idx), axis=None))

        # print(time.time() - a)
    elif get_special_sample == "covid_uniform":
        a = time.time()
        # if True:
        #     labels = [dataset[indice][1] for indice in indices]
        #     with open('shuffled_labels.txt', 'w') as filehandle:
        #         for label in labels:
        #             filehandle.write('%s\n' % label)
        if True:
            labels = []
            with open('shuffled_labels.txt', 'r') as filehandle:
                for line in filehandle:
                    current_place = line[:-1]
                    labels.append(int(current_place))

        labels = np.asarray(labels)
        covid_labels = labels == COVID_LABEL
        train_covid_msk, valid_covid_msk = covid_labels[:split], covid_labels[split:]
        train_covid_idx, valid_covid_idx = train_idx[train_covid_msk], valid_idx[valid_covid_msk]
        nb_covid_train, nb_covid_valid = sum(train_covid_msk), sum(valid_covid_msk)

        normal_labels = labels == NORMAL_LABEL
        train_normal_msk, valid_normal_msk = normal_labels[:split], normal_labels[split:]
        train_normal_idx, valid_normal_idx = train_idx[train_normal_msk], valid_idx[valid_normal_msk]
        # train_normal_cov_idx, valid_normal_cov_idx = train_normal_idx[-nb_covid_train:], valid_normal_idx[-nb_covid_valid:]
        train_normal_cov_idx, valid_normal_cov_idx = train_normal_idx[-nb_covid_train:], valid_normal_idx

        pneu_labels = labels == PNEUMONIA_LABEL
        train_pneu_msk, valid_pneu_msk = pneu_labels[:split], pneu_labels[split:]
        train_pneu_idx, valid_pneu_idx = train_idx[train_pneu_msk], valid_idx[valid_pneu_msk]
        # train_pneu_cov_idx, valid_pneu_cov_idx = train_pneu_idx[-nb_covid_train:], valid_pneu_idx[-nb_covid_valid:]
        train_pneu_cov_idx, valid_pneu_cov_idx = train_pneu_idx[-nb_covid_train:], valid_pneu_idx

        train_sampler = SubsetRandomSampler(np.concatenate((train_covid_idx, train_normal_cov_idx, train_pneu_cov_idx), axis=None))
        valid_sampler = SubsetRandomSampler(np.concatenate((valid_covid_idx, valid_normal_cov_idx, valid_pneu_cov_idx), axis=None))

        # print(time.time() - a)
    else:
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(dataset,
                    batch_size=batch_size, sampler=train_sampler)
    valid_loader = torch.utils.data.DataLoader(dataset,
                    batch_size=batch_size, sampler=valid_sampler)

    return train_loader, valid_loader


class SpiralDataset(torch.utils.data.Dataset):

    def __init__(self, n_points=1000, noise=0.2):
        self.points = torch.Tensor(n_points, 7)
        self.labels = torch.LongTensor(n_points)

        n_positive = n_points // 2
        n_negative = n_points = n_positive

        for i, point in enumerate(self._gen_spiral_points(n_positive, 0, noise)):
            self.points[i], self.labels[i] = point, 1

        for i, point in enumerate(self._gen_spiral_points(n_negative, math.pi, noise)):
            self.points[i+n_positive] = point
            self.labels[i+n_positive] = 0


    def _gen_spiral_points(self, n_points, delta_t, noise):
        for i in range(n_points):
            r = i / n_points * 5
            t = 1.75 * i / n_points * 2 * math.pi + delta_t
            x = r * math.sin(t) + random.uniform(-1, 1) * noise
            y = r * math.cos(t) + random.uniform(-1, 1) * noise
            yield torch.Tensor([x, y, x**2, y**2, x*y, math.sin(x), math.sin(y)])


    def __len__(self):
        return len(self.labels)


    def __getitem__(self, i):
        return self.points[i], self.labels[i]


    def to_numpy(self):
        return self.points.numpy(), self.labels.numpy()


if __name__ == '__main__':
    mnist = load_mnist(download=True)
    cifar = load_cifar10(download=True)
