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

    def __init__(self, csv_file, root_dir, transform=None, train=True, use_features=True, three_classes=True, features_len=100):
        """
        Args:
            csv_file (DataFrame): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.videos_name = csv_file
        self.root_dir = root_dir
        self.transform = transform
        self.train = train
        self.use_features = use_features
        self.three_classes = {'forehand': 0, 'backhand': 1, 'service': 2, 'smash': 2}
        self.features_len = features_len

    def __len__(self):
        return len(self.videos_name)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        label = 0
        for class_name, class_id in self.three_classes.items():
            if class_name in self.videos_name.iloc[idx, 0]:
                label = class_id
                break

        video_path = os.path.join(self.root_dir, self.videos_name.iloc[idx, 0], self.videos_name.iloc[idx, 1])
        features_path = os.path.splitext(video_path)[0] + '.csv'

        sample = {'gt': label,
                  'vid_folder': self.videos_name.iloc[idx, 0], 'vid_name': self.videos_name.iloc[idx, 1]}
        if not self.use_features:
            vid_frames = video_to_frames(video_path)

            if self.transform:
                frames = []
                for frame in vid_frames:
                    frame = self.transform(frame)
                    frames.append(frame)

                vid_frames = torch.stack(frames)
            sample['frames'] = vid_frames
        else:
            vid_features = pd.read_csv(features_path)
            diff = self.features_len - len(vid_features)
            '''if diff > 0:
                zeros_df = pd.DataFrame(np.zeros((diff, len(vid_features.columns))), columns=vid_features.columns)
                vid_features = vid_features.append(zeros_df, ignore_index=True)
                # vid_frames = torch.cat([vid_frames, torch.zeros((diff, *vid_frames[0].size()))])
            if diff < 0:
                vid_features = vid_features.iloc[:100, :]
                # vid_frames = vid_frames[:100, :, :, :]'''
            sample['features'] = torch.Tensor(vid_features.values)
            #sample['frames'] = vid_frames
        return sample


class StrokesDataset(Dataset):
    """ Strokes dataset."""

    def __init__(self, csv_file, root_dir, transform=None, train=True, use_features=True, y_full=0):
        """
        Args:
            csv_file (DataFrame): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.train = train
        self.use_features = use_features
        self.three_classes = {'forehand': 0, 'backhand': 1, 'service': 2, 'smash': 2}
        self.y_full = y_full

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        label = self.df.iloc[idx, 1]

        video_path = os.path.join(self.root_dir, self.df.iloc[idx, 0])
        features_path = os.path.splitext(video_path)[0] + '.csv'

        sample = {'gt': label, 'vid_name': self.df.iloc[idx, 0]}
        if not self.use_features:
            vid_frames = video_to_frames(video_path)

            if self.transform:
                frames = []
                for frame in vid_frames:
                    frame = self.transform(frame)
                    frames.append(frame)

                vid_frames = torch.stack(frames)
            sample['frames'] = vid_frames
        else:
            vid_features = pd.read_csv(features_path)
            sample['features'] = torch.Tensor(vid_features.values)
            if self.y_full == 1:
                label = np.array([label] * vid_features.shape[0])
                sample['gt'] = label
            elif self.y_full == 2:
                labels_size = vid_features.shape[0] * 3 // 4
                label = np.array([3] * (vid_features.shape[0] - labels_size) + [label] * labels_size)
                sample['gt'] = label
            elif self.y_full == 3:
                labels_size = vid_features.shape[0] // 2
                label = np.array([label] * labels_size)
                sample['gt'] = label

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


def create_train_valid_test_datasets(csv_file, root_dir, transform=None):
    videos_name = pd.read_csv(csv_file)
    test_player_id = 40
    test_videos_name = videos_name[
        videos_name.loc[:, 'name'].str.contains(f'p{test_player_id}', na=False)]
    remaining_ids = list(range(1, 55))
    remaining_ids.remove(test_player_id)
    valid_ids = np.random.choice(remaining_ids, 5, replace=False)
    mask = videos_name.loc[:, 'name'].str.contains('|'.join([f'p{id}' for id in valid_ids]), na=False)
    valid_videos_name = videos_name[mask]
    train_videos = videos_name.drop(index=test_videos_name.index.union(valid_videos_name.index))
    train_ds = ThetisDataset(train_videos, root_dir, transform=transform)
    valid_ds = ThetisDataset(valid_videos_name, root_dir, transform=transform)
    test_ds = ThetisDataset(test_videos_name, root_dir, transform=transform)
    return train_ds, valid_ds, test_ds


def get_dataloaders(csv_file, root_dir, transform, batch_size):
    ds = StrokesDataset(csv_file=csv_file, root_dir=root_dir, transform=transform, train=True, use_features=True)
    length = len(ds)
    train_size = int(0.85 * length)
    train_ds, valid_ds = torch.utils.data.random_split(ds, (train_size, length - train_size))
    print(f'train set size is : {train_size}')
    print(f'validation set size is : {length - train_size}')

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_dl = DataLoader(valid_ds, batch_size=batch_size, shuffle=True)
    return train_dl, valid_dl


if __name__ == '__main__':
    train,valid,test = create_train_valid_test_datasets('../dataset/THETIS/VIDEO_RGB/THETIS_data.csv', '../dataset/THETIS/VIDEO_RGB/')

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