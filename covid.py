import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from skimage import io
from skimage.color import gray2rgb
from torch import optim
from torch.utils.data import Dataset
from torchvision import models

from const import COVID_LABEL, NORMAL_LABEL, PNEUMONIA_LABEL
from deeplib import datasets
from deeplib.training import train, test

train_csv = pd.read_csv("dataset/data-extractor/train_split_v3.txt", header=None, sep=" ", usecols=range(0, 4))
train_dir = r"dataset/data-extractor/data/train/"
test_csv = pd.read_csv("dataset/data-extractor/test_split_v3.txt", header=None, sep=" ", usecols=range(0, 4))
test_dir = r"dataset/data-extractor/data/test/"

FREEZES = [0,5,6,7,8]


class CovidDataset(Dataset):
    def __init__(self, dataframe, datadir):
        self.length = len(dataframe)
        self.file_names = dataframe[1]
        self.labels = dataframe[2]
        self.data = dataframe
        self.datadir = datadir
        self.target_transform = None
        self.transform = None
        self.label_no = {"normal": NORMAL_LABEL, "pneumonia": PNEUMONIA_LABEL, "COVID-19": COVID_LABEL}

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        im = io.imread(self.datadir + self.file_names[idx])
        im = gray2rgb(im)

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


def main():
    # todo tomorrow
    # if False:
    #     datasets.get_special_sample = "covid_uniform"
    #     resnet18 = models.resnet18(pretrained=True)
    #     run_model("uniform_small_18", resnet18, 0.00005, 10)
    #
    # if False:
    #     datasets.get_special_sample = "covid_uniform"
    #     resnet34 = models.resnet34(pretrained=True)
    #     run_model("uniform_small_34", resnet34, 0.00005, 10)

    if False:
        datasets.get_special_sample = "covid_uniform"
        resnet50 = models.resnet50(pretrained=True)
        run_model("uniform_small_50", resnet50, 0.00005, 10)

    # if False:
    #     datasets.get_special_sample = "covid_uniform"
    #     resnet101 = models.resnet101(pretrained=True)
    #     run_model("uniform_small_101", resnet101, 0.00005, 10, batch_size=16)
    #
    # if False:
    #     datasets.get_special_sample = "covid_uniform"
    #     resnet152 = models.resnet152(pretrained=True)
    #     run_model("uniform_small_152", resnet152, 0.00005, 10, batch_size=16)
    #
    #
    # if False:
    #     datasets.get_special_sample = None
    #     resnet101 = models.resnet101(pretrained=True)
    #     run_model("base_101", resnet101, 0.00005, 10, batch_size=16)
    #
    #     datasets.get_special_sample = None
    #     resnet152 = models.resnet152(pretrained=True)
    #     run_model("base_152", resnet152, 0.00005, 10, batch_size=16)
    #
    # nocav_name = "nocov_resnet50"
    # if False:
    #     datasets.get_special_sample = "no_covid"
    #     model = models.resnet50(pretrained=True)
    #     run_model(nocav_name, model, 0.000005, 10)

    # datasets.get_special_sample = "no_covid"
    # resnet18 = models.resnet18()
    # resnet18.load_state_dict(torch.load(nocav_name))

    # run_model("base_resnet34", models.resnet34(pretrained=True), )

    # datasets.get_special_sample = "no_covid"
    # resnet50 = models.resnet50(pretrained=True)
    # run_model("nocov_resnet50", resnet50, 0.000005, 10)

    if True:
        for fr in FREEZES:
            print(fr)
            datasets.get_special_sample = "covid_uniform"

            # todo no load
            model = models.resnet50()
            model.load_state_dict(torch.load("modelno_covid_nocov_resnet50"))

            ct = 0
            for child in model.children():
                if ct < fr:
                    for param in child.parameters():
                        param.requires_grad = False
                ct += 1
            run_model("pretrained_resnet50", model, 0.00005, 10)
            # model.eval()

    # if True:
    #     datasets.get_special_sample = "no_covid"
    #     model = models.resnet101(pretrained=True)
    #     run_model('nocov_101', model, 0.000005, 10, batch_size=16)
    #
    # if True:
    #     datasets.get_special_sample = "covid_uniform"
    #
    #     # todo no load
    #     # model = models.resnet50()
    #     # model.load_state_dict(torch.load("no_covid_nocov_resnet50"))
    #
    #     ct = 0
    #     for child in model.children():
    #         ct += 1
    #         if ct < 0:
    #             for param in child.parameters():
    #                 param.requires_grad = False
    #     run_model("pretrained_resnet101", model, 0.00005, 10)
    #
    # if True:
    #     datasets.get_special_sample = "no_covid"
    #     model = models.resnet152(pretrained=True)
    #     run_model('nocov_152', model, 0.000005, 10, batch_size=16)
    #
    # if True:
    #     datasets.get_special_sample = "covid_uniform"
    #
    #     # todo no load
    #     # model = models.resnet50()
    #     # model.load_state_dict(torch.load("no_covid_nocov_resnet50"))
    #
    #     ct = 0
    #     for child in model.children():
    #         ct += 1
    #         if ct < 0:
    #             for param in child.parameters():
    #                 param.requires_grad = False
    #     run_model("pretrained_resnet152", model, 0.00005, 10)


def run_model(name, model, learning_rate, n_epoch, batch_size = 32):
    train_dataset = CovidDataset(train_csv, train_dir)
    test_dataset = CovidDataset(test_csv, test_dir)
    # display(train_dataset)
    model.cuda()
    use_gpu = True
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    history = train(model, optimizer, train_dataset, n_epoch, batch_size, use_gpu=use_gpu)
    history.display()
    with open("./history" + datasets.get_special_sample + "_" + name, "wb") as f:
        pickle.dump(history, f)
    model = history.history['model'][1]
    try:
        test_acc, test_loss, covid_recall, covid_accuracy = test(model, test_dataset, batch_size, use_gpu=use_gpu)
        f1 = 2 * covid_recall * covid_accuracy / (covid_recall + covid_accuracy)
        print('Test:\n\tLoss: {}\n\tAccuracy: {}'.format(test_loss, test_acc))
        print('Test recall on Covid: {:.2f} - Test precision on Covid: {:.2f} - Test f1 score: {:.2f}'.format(covid_recall, covid_accuracy, f1))
        print("")
    except:
        print("STUFF HAPPENS YO")
        print("")
    torch.save(history.history['model'][1].state_dict(), "./model" + datasets.get_special_sample + "_" + name)


if __name__ == "__main__":
    main()

