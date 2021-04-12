from collections import deque
import numpy as np
import argparse
import imutils
import cv2
import torch
import torchvision
import torch.nn as nn
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import find_peaks

from src.ball_tracker_net import BallTrackerNet
from src.court_detection import CourtDetector
from src.detection import center_of_box
from src.utils import get_video_properties


def combine_three_frames(frame1, frame2, frame3, width, height):
    img = cv2.resize(frame1, (width, height))
    # input must be float type
    img = img.astype(np.float32)

    # resize it
    img1 = cv2.resize(frame2, (width, height))
    # input must be float type
    img1 = img1.astype(np.float32)

    # resize it
    img2 = cv2.resize(frame3, (width, height))
    # input must be float type
    img2 = img2.astype(np.float32)

    # combine three imgs to  (width , height, rgb*3)
    imgs = np.concatenate((img, img1, img2), axis=2)

    # since the odering of TrackNet  is 'channels_first', so we need to change the axis
    imgs = np.rollaxis(imgs, 2, 0)
    return np.array(imgs)


class BallDetector:
    def __init__(self, save_state, out_channels=2):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.detector = BallTrackerNet(out_channels=out_channels)
        saved_state_dict = torch.load(save_state)
        self.detector.load_state_dict(saved_state_dict['model_state'])
        self.detector.eval().to(self.device)

        self.current_frame = None
        self.last_frame = None
        self.before_last_frame = None

        self.video_width = None
        self.video_height = None
        self.model_input_width = 640
        self.model_input_height = 360

        self.threshold_dist = 100
        self.xy_coordinates = np.array([[None, None], [None, None]])

    def detect_ball(self, frame):
        if self.video_width is None:
            self.video_width = frame.shape[1]
            self.video_height = frame.shape[0]
        self.last_frame = self.before_last_frame
        self.before_last_frame = self.current_frame
        self.current_frame = frame.copy()
        if self.last_frame is not None:
            frames = combine_three_frames(self.current_frame, self.before_last_frame, self.last_frame,
                                          self.model_input_width, self.model_input_height)
            frames = (torch.from_numpy(frames) / 255).to(self.device)
            x, y = self.detector.inference(frames)
            if x is not None:
                x = x * (self.video_width / self.model_input_width)
                y = y * (self.video_height / self.model_input_height)
                if self.xy_coordinates[-1][0] is not None:
                    if np.linalg.norm(np.array([x,y]) - self.xy_coordinates[-1]) > self.threshold_dist:
                        x, y = None, None
            self.xy_coordinates = np.append(self.xy_coordinates, np.array([[x, y]]), axis=0)

    def mark_positions(self, frame, mark_num=4, frame_num=None, ball_color='yellow'):
        if frame_num is not None:
            q = self.xy_coordinates[frame_num-mark_num+1:frame_num+1, :]
        else:
            q = self.xy_coordinates[-mark_num:, :]
        pil_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(pil_image)
        for i in range(q.shape[0]):
            if q[i, 0] is not None:
                draw_x = q[i, 0]
                draw_y = q[i, 1]
                bbox = (draw_x - 2, draw_y - 2, draw_x + 2, draw_y + 2)
                draw = ImageDraw.Draw(pil_image)
                draw.ellipse(bbox, outline=ball_color)

            # Convert PIL image format back to opencv image format
            frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        return frame

    def show_y_graph(self, player_1_boxes, player_2_boxes):
        player_1_centers = np.array([center_of_box(box) for box in player_1_boxes])
        player_1_y_values = player_1_centers[:, 1]
        player_1_y_values -= np.array([(box[3] - box[1]) // 4 for box in player_1_boxes])
        player_2_centers = []
        for box in player_2_boxes:
            if box[0] is not None:
                player_2_centers.append(center_of_box(box))
            else:
                player_2_centers.append([None, None])
        player_2_centers = np.array(player_2_centers)
        player_2_y_values = player_2_centers[:, 1]

        y_values = self.xy_coordinates[:, 1].copy()
        x_values = self.xy_coordinates[:, 0].copy()

        '''outliers_y = self._get_outliers_indices(y_values)

        outliers_x = self._get_outliers_indices(x_values)
        outliers = outliers_x + outliers_y
        print(f'Outliers : {outliers}')
        y_values[outliers] = None'''
        plt.figure()
        plt.scatter(range(len(y_values)), y_values)
        plt.plot(range(len(player_1_y_values)), player_1_y_values, color='r')
        plt.plot(range(len(player_2_y_values)), player_2_y_values, color='g')
        plt.show()

    def _get_outliers_indices(self, values, max_abs_diff=50):
        values = values.copy()
        values[values == None] = -1
        diff = np.array([val1 - val2 for val1, val2 in zip(values[1:], values)])
        outliers = []
        for i, (val1, val2) in enumerate(zip(diff, diff[1:])):
            if abs(val1) > max_abs_diff and abs(val2) > max_abs_diff:
                outliers.append(i + 1)
        none_indices = [i for i, y in enumerate(values) if y == -1]
        outliers = [i for i in outliers if i not in none_indices]

        return outliers

    def get_max_value_frames(self, window_len=60):
        y_values = self.xy_coordinates[:, 1]
        y_values[np.where(np.array(y_values) == None)[0]] = np.nan
        max_indices = []
        for i in range(0, len(y_values) - window_len, window_len / 2):
            window = y_values[i:i + window_len]
            indices = np.nanargmax(window) + i
            max_indices.append(indices)
        max_indices = list(dict.fromkeys(max_indices))
        return max_indices


if __name__ == "__main__":
    ball_detector = BallDetector('saved states/tracknet_weights_lr_1.0_epochs_150_last_trained.pth')
    cap = cv2.VideoCapture('../videos/vid1.mp4')
    # get videos properties
    fps, length, v_width, v_height = get_video_properties(cap)

    frame_i = 0
    while True:
        ret, frame = cap.read()
        frame_i += 1
        if not ret:
            break

        ball_detector.detect_ball(frame)


    cap.release()
    cv2.destroyAllWindows()

    from scipy.interpolate import interp1d

    y_values = ball_detector.xy_coordinates[:,1]

    new = signal.savgol_filter(y_values, 3, 2)

    x = np.arange(0, len(new))
    indices = [i for i, val in enumerate(new) if np.isnan(val)]
    x = np.delete(x, indices)
    y = np.delete(new, indices)
    f = interp1d(x, y, fill_value="extrapolate")
    f2 = interp1d(x, y, kind='cubic', fill_value="extrapolate")
    xnew = np.linspace(0, len(y_values), num=len(y_values), endpoint=True)
    plt.plot(np.arange(0, len(new)), new, 'o',xnew,
             f2(xnew), '-r')
    plt.legend(['data', 'inter'], loc='best')
    plt.show()

    positions = f2(xnew)
    peaks, _ = find_peaks(positions, distance=30)
    a = np.diff(peaks)
    plt.plot(positions)
    plt.plot(peaks, positions[peaks], "x")
    plt.show()