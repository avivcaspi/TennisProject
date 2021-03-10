import os

import torch
import torchvision
import numpy as np
import cv2
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn as nn
from torchvision.transforms import ToTensor

from src.datasets import ThetisDataset
from src.trainer import Trainer
from utils import get_dtype
import pandas as pd


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = torchvision.models.inception_v3(pretrained=True)
        self.feature_extractor.fc = Identity()

    def forward(self, x):
        output = self.feature_extractor(x)
        return output


class ActionRecognition(nn.Module):
    def __init__(self, num_classes, input_size=2048, num_layers=3, hidden_size=90, dtype=torch.cuda.FloatTensor):
        super().__init__()
        self.dtype = dtype
        self.input_size = input_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.LSTM = nn.LSTM(input_size, hidden_size, num_layers, bias=True, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x shape is (batch_size, seq_len, input_size)
        h0, c0 = self.init_state(x.size(0))
        output, (hn, cn) = self.LSTM(x, (h0, c0))
        output = output[:, -1, :]
        scores = self.fc(output)
        # scores shape is (batch_size, num_classes)
        return scores

    def init_state(self, batch_size):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size).type(self.dtype),
                torch.zeros(self.num_layers, batch_size, self.hidden_size).type(self.dtype))


def create_features_from_vids():
    dtype = get_dtype()
    feature_extractor = FeatureExtractor()
    feature_extractor.eval()
    feature_extractor.type(dtype)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    dataset = ThetisDataset('../dataset/THETIS/VIDEO_RGB/THETIS_data.csv', '../dataset/THETIS/VIDEO_RGB/',
                            transform=transforms.Compose([ToTensor(), normalize]))
    batch_size = 32
    count = 0
    for vid in dataset:
        count += 1
        frames = vid['frames']
        print(len(frames))

        features = []
        for batch in frames.split(batch_size):
            batch = batch.type(dtype)
            with torch.no_grad():
                # forward pass
                batch_features = feature_extractor(batch)
                features.append(batch_features.cpu().numpy())

        df = pd.DataFrame(np.concatenate(features, axis=0))

        outfile_path = os.path.join('../dataset/THETIS/VIDEO_RGB/', vid['vid_folder'],
                                    os.path.splitext(vid['vid_name'])[0] + '.csv')
        df.to_csv(outfile_path, index=False)

        print(count)


if __name__ == "__main__":
    dtype = get_dtype()
    #feature_extractor = FeatureExtractor()
    model = ActionRecognition(3, dtype=dtype)
    #feature_extractor.eval()

    #feature_extractor.type(dtype)
    model.type(dtype)
    batch_size = 16

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    dataset = ThetisDataset('../dataset/THETIS/VIDEO_RGB/THETIS_data.csv', '../dataset/THETIS/VIDEO_RGB/',
                            transform=transforms.Compose([ToTensor(), normalize]), use_features=True)
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    trainer = Trainer(model, dl)
    trainer.train(50)
    '''batch = None
    video = cv2.VideoCapture('../videos/vid1.mp4')
    while True:
        ret, frame = video.read()
        if ret:
            frame_t = frame.transpose((2, 0, 1)) / 255
            frame_tensor = torch.from_numpy(frame_t).type(dtype)
            frame_tensor = normalize(frame_tensor).unsqueeze(0)
            with torch.no_grad():
                # forward pass
                features = feature_extractor(frame_tensor)
            features = features.unsqueeze(1)
            if batch is None:
                batch = features
            else:
                batch = torch.cat([batch, features], dim=1)
            if batch.size(1) > seq_len:
                # TODO this might be problem, need to get the vector out of gpu
                remove = batch[:,0,:]
                remove.detach().cpu()
                batch = batch[:, 1:, :]
                output = model(batch)
        else:
            break
    video.release()

    cv2.destroyAllWindows()'''

