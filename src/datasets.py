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

    def __init__(self, csv_file, root_dir, transform=None, train=True, use_features=True, three_classes=True,
                 features_len=100):
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
            # sample['frames'] = vid_frames
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
                labels_size = vid_features.shape[0] // 4
                label = np.array([label] * labels_size)
                sample['gt'] = label

        return sample


def getInputArr(path, path1, path2, width, height):
    try:
        # read the image
        img = cv2.imread(path, 1)
        # resize it
        img = cv2.resize(img, (width, height))
        # input must be float type
        img = img.astype(np.float32)

        # read the image
        img1 = cv2.imread(path1, 1)
        # resize it
        img1 = cv2.resize(img1, (width, height))
        # input must be float type
        img1 = img1.astype(np.float32)

        # read the image
        img2 = cv2.imread(path2, 1)
        # resize it
        img2 = cv2.resize(img2, (width, height))
        # input must be float type
        img2 = img2.astype(np.float32)

        # combine three imgs to  (width , height, rgb*3)
        imgs = np.concatenate((img, img1, img2), axis=2)

        # since the odering of TrackNet  is 'channels_first', so we need to change the axis
        imgs = np.rollaxis(imgs, 2, 0)
        return np.array(imgs)

    except Exception as e:

        print(path, e)


def getOutputArr(path, num_classes, width, height):
    seg_labels = np.zeros((height, width, num_classes))
    try:
        img = cv2.imread(path, 1)
        img = cv2.resize(img, (width, height))
        img = img[:, :, 0]

        for c in range(num_classes):
            seg_labels[:, :, c] = (img == c).astype(int)

    except Exception as e:
        print(e)

    seg_labels = np.reshape(seg_labels, (width * height, num_classes))
    seg_labels = seg_labels.transpose([1, 0]).argmax(0)
    return np.array(seg_labels)


class TrackNetDataset(Dataset):
    """ TrackNet dataset."""

    def __init__(self, csv_file, transform=None, train=True, input_height=360, input_width=640,
                 output_height=360, output_width=640, num_classes=256):
        """
        Args:
            csv_file (DataFrame): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.df = pd.read_csv(csv_file)
        self.transform = transform
        self.train = train
        self.input_height = input_height
        self.input_width = input_width
        self.output_height = output_height
        self.output_width = output_width
        self.num_classes = num_classes

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        path1, path2, path3, gt_path, x, y, status = self.df.iloc[idx, :]
        if np.math.isnan(x):
            x = -1
            y = -1

        if np.math.isnan(status):
            status = -1

        vid_frames = getInputArr(path1, path2, path3, self.input_width, self.input_height)

        gt_path = gt_path.replace("groundtruth", f"groundtruth_{self.num_classes}")
        gt = getOutputArr(gt_path, self.num_classes, self.output_width, self.output_height)

        vid_frames = torch.from_numpy(vid_frames) / 255
        gt = torch.from_numpy(gt)

        sample = {'frames': vid_frames, 'gt': gt, 'gt_path': gt_path, 'x_true': x, 'y_true': y, 'status': status}

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
    """
    Split Thetis dataset into train validation and test sets
    """
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


def get_dataloaders(csv_file, root_dir, transform, batch_size, dataset_type='stroke', num_classes=256, num_workers=0, seed=42):
    """
    Get train and validation dataloader for strokes and tracknet datasets
    """
    ds = []
    if dataset_type == 'stroke':
        ds = StrokesDataset(csv_file=csv_file, root_dir=root_dir, transform=transform, train=True, use_features=True)
    elif dataset_type == 'tracknet':
        ds = TrackNetDataset(csv_file=csv_file, train=True, num_classes=num_classes)
    length = len(ds)
    train_size = int(0.85 * length)
    train_ds, valid_ds = torch.utils.data.random_split(ds, (train_size, length - train_size),
                                                       generator=torch.Generator().manual_seed(seed))
    print(f'train set size is : {train_size}')
    print(f'validation set size is : {length - train_size}')

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_dl = DataLoader(valid_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return train_dl, valid_dl


if __name__ == '__main__':
    train_dl, _ = get_dataloaders('../dataset/Dataset/training_model2.csv', root_dir=None, transform=None,
                                  batch_size=1, dataset_type='tracknet', num_workers=4)

    import time

    s = time.time()

    for i, a in enumerate(train_dl):
        print(a['gt_path'])
        if i == 100:
            break
    print(time.time() - s)

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
