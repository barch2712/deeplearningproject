from math import ceil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torchvision
from PIL import Image
from matplotlib import cm
from skimage.transform import resize
from torch import optim, nn
from torch.nn.functional import pad
from torch.utils.data import Dataset
from skimage import io, transform
from skimage.color import rgb2gray, gray2rgb
from torchvision import models, transforms
from torchvision.transforms.functional import center_crop

from deeplib.training import train, test

train_csv = pd.read_csv("dataset/data-extractor/train_split_v3.txt", header=None, sep=" ", usecols=range(0, 4))
train_dir = r"dataset/data-extractor/data/train/"
test_csv = pd.read_csv("dataset/data-extractor/test_split_v3.txt", header=None, sep=" ", usecols=range(0, 4))
test_dir = r"dataset/data-extractor/data/test/"


class CovidDataset(Dataset):
    def __init__(self, dataframe, datadir):
        self.length = len(dataframe)
        self.file_names = dataframe[1]
        self.labels = dataframe[2]
        self.data = dataframe
        self.datadir = datadir
        self.target_transform = None
        self.transform = None
        self.label_no = {"normal": 0, "pneumonia": 1, "COVID-19": 2}

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        im = io.imread(self.datadir + self.file_names[idx])
        im = gray2rgb(im)

        # im = Image.open(self.datadir + self.file_names[idx])

        # im = Image.fromarray(im)
        # im = transforms.Resize(224, 224)(im)

        # wanted_size = [224, 224]
        # if im.shape[0] < wanted_size[0] or im.shape[1] < wanted_size[1]:
        #     im = pad(im, [ceil((wanted_size[0] - im.shape[0])/2), ceil((wanted_size[1] - im.shape[1])/2)])
        # center_crop(im, wanted_size)
        # im = resize(im, wanted_size, interpolation=2)

        # im = transforms.ToTensor()(im)
        # im = im[:wanted_size, :wanted_size]

        # im = np.moveaxis(im, 0, -1)
        # if im.shape[1] < wanted_size or im.shape[2] < wanted_size:
        #     wanted_size
        #     im = transforms.Pad(())

        # im = Image.fromarray(im.numpy(), mode='L')
        im = im[:, :, :3]
        im = Image.fromarray(im)

        if self.transform is not None:
            im = self.transform(im)

        label = self.label_no[self.labels[idx]]
        if self.target_transform is not None:
            label = self.target_transform(label)

        return im, label


def display(train_val):
    fig, axes = plt.subplots(6, 6, figsize=(20, 20))
    axes = axes.flatten()
    for ax in axes:
        im, label = train_val[np.random.randint(len(train_val))]
        ax.imshow(im, cmap="gray")
        ax.set_title(str(label) + " " + str(im.shape))
    plt.show()


# class MyResNet(nn.Module):
#
#     def __init__(self, in_channels=1):
#         super(MyResNet, self).__init__()
#
#         # bring resnet
#         self.model = torchvision.models.resnet18()
#
#         # original definition of the first layer on the renset class
#         # self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
#
#         # your case
#         self.model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
#
#     def forward(self, x):
#         return self.model(x)


def main():
    train_dataset = CovidDataset(train_csv, train_dir)
    test_dataset = CovidDataset(test_csv, test_dir)

    # display(train_dataset)

    resnet18 = models.resnet18(pretrained=True)
    resnet18.cuda()

    use_gpu = True
    n_epoch = 0
    batch_size = 32
    learning_rate = 0.005
    optimizer = optim.Adam(resnet18.parameters(), lr=learning_rate)

    history = train(resnet18, optimizer, train_dataset, n_epoch, batch_size, use_gpu=use_gpu)

    history.display()

    test_acc, test_loss = test(resnet18, test_dataset, batch_size, use_gpu=use_gpu)
    print('Test:\n\tLoss: {}\n\tAccuracy: {}'.format(test_loss, test_acc))


if __name__ == "__main__":
    main()

