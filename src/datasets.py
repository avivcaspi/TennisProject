import cv2
import torch
from torch.utils.data import Dataset
import pandas as pd
from skimage import io, transform, color
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import os
from sklearn.model_selection import train_test_split


class ThetisDataset(Dataset):
    """ THETIS dataset."""

    def __init__(self, csv_file, root_dir, transform=None, train=True):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.patches_name = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.train = train

    def __len__(self):
        return len(self.patches_name)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        train_test_str = 'train' if self.train else 'test'
        label = None

        green_img_name = os.path.join(self.root_dir,
                                      green_str + self.patches_name.iloc[idx, 0] + '.TIF')

        if self.train:
            gt_img_name = os.path.join(self.root_dir,
                                       gt_str + self.patches_name.iloc[idx, 0] + '.TIF')
            gt_img = io.imread(gt_img_name) / 255

        red_img = np.expand_dims(io.imread(red_img_name) / 65535, axis=-1)
        green_img = np.expand_dims(io.imread(green_img_name) / 65535, axis=-1)
        blue_img = np.expand_dims(io.imread(blue_img_name) / 65535, axis=-1)

        image = np.concatenate([red_img, green_img, blue_img], axis=-1)

        sample = {'image': image, 'gt': gt_img, 'patch_name': self.patches_name.iloc[idx, 0]}

        if self.transform:
            sample = self.transform(sample)

        return sample

    if __name__ == '__main__':
        dataset = ThetisDataset('../dataset/THETIS/VIDEO_RGB/THETIS_data.csv', '../dataset/THETIS/VIDEO_RGB/',
                                  transform=transforms.Compose([Rescale(192), ToTensor()]), weakly=True, train=False)
        print(len(dataset))

