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

    def __init__(self, csv_file, root_dir, transform=None, train=True, features=True, three_classes=True):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.videos_name = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.train = train
        self.features = features
        self.three_classes = {'forehand': 0, 'backhand': 1, 'service': 2, 'smash': 2}

    def __len__(self):
        return len(self.videos_name)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        train_test_str = 'train' if self.train else 'test'
        label = 0
        for class_name, class_id in self.three_classes.items():
            if class_name in self.videos_name.iloc[idx, 0]:
                label = class_id
                break

        video_path = os.path.join(self.root_dir, self.videos_name.iloc[idx, 0], self.videos_name.iloc[idx, 1])
        features_path = os.path.splitext(video_path)[0] + '.csv'

        vid_frames = video_to_frames(video_path)

        if self.transform:
            frames = []
            for frame in vid_frames:
                frame = self.transform(frame)
                frames.append(frame)

            vid_frames = torch.stack(frames)
        sample = {'frames': vid_frames, 'gt': label,
                  'vid_folder': self.videos_name.iloc[idx, 0], 'vid_name': self.videos_name.iloc[idx, 1]}
        if self.features:
            vid_features = pd.read_csv(features_path)
            sample['features'] = vid_features
        return sample


def video_to_frames(video_filename):
    """Extract frames from video"""
    cap = cv2.VideoCapture(video_filename)
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    frames = []
    if cap.isOpened() and video_length > 0:
        count = 0
        success, image = cap.read()
        while success:
            frames.append(image)
            success, image = cap.read()
            count += 1
    cap.release()
    return np.array(frames)


if __name__ == '__main__':
    dataset = ThetisDataset('../dataset/THETIS/VIDEO_RGB/THETIS_data.csv', '../dataset/THETIS/VIDEO_RGB/')
    print(len(dataset))
    vid = dataset[0]
    a = 0
    '''rootdir = '../dataset/THETIS/VIDEO_RGB/'
    data = []
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            data.append([os.path.split(subdir)[-1], file])
            print([os.path.split(subdir)[-1], file])
    df = pd.DataFrame(data, columns=['folder', 'name'])
    outfile_path = os.path.join(rootdir, 'THETIS_data.csv')
    df.to_csv(outfile_path, index=False)
'''