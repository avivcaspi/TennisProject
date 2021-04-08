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
from src.ball_tracker_net import BallTrackerNet
from src.court_detection import CourtDetector
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
        self.detector = BallTrackerNet(out_channels=2)
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
            self.xy_coordinates = np.append(self.xy_coordinates, np.array([[x, y]]), axis=0)

    def mark_positions(self, frame, mark_num=4):
        q = self.xy_coordinates[-mark_num:, :]
        pil_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(pil_image)
        for i in range(q.shape[0]):
            if q[i, 0] is not None:
                draw_x = q[i, 0]
                draw_y = q[i, 1]
                bbox = (draw_x - 2, draw_y - 2, draw_x + 2, draw_y + 2)
                draw = ImageDraw.Draw(pil_image)
                draw.ellipse(bbox, outline='yellow')

            # Convert PIL image format back to opencv image format
            frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        return frame

    def show_y_graph(self):
        y_values = self.xy_coordinates[:, 1]
        plt.figure()
        plt.plot(range(len(y_values)), y_values)

        plt.show()

    def get_max_value_frames(self, window_len=60):
        y_values = self.xy_coordinates[:, 1]
        y_values[np.where(np.array(y_values) == None)[0]] = np.nan
        max_indices = []
        for i in range(0, len(y_values) - window_len, window_len/2):
            window = y_values[i:i+window_len]
            indices = np.nanargmax(window) + i
            max_indices.append(indices)
        max_indices = list(dict.fromkeys(max_indices))
        return max_indices


if __name__ == "__main__":
    ball_detector = BallDetector('saved states/tracknet_weights_lr_0.05_epochs_280.pth')
    cap = cv2.VideoCapture('../videos/vid7.mp4')
    # get videos properties
    fps, length, v_width, v_height = get_video_properties(cap)

    frame_i = 0
    while True:
        ret, frame = cap.read()
        frame_i += 1
        if not ret:
            break

        ball_detector.detect_ball(frame)

    max_value_indices = ball_detector.get_max_value_frames()
    ball_detector.show_y_graph(max_value_indices)



    cap.release()

    cv2.destroyAllWindows()

    for index in max_value_indices:
        cap = cv2.VideoCapture('../videos/vid7.mp4')
        # get videos properties
        fps, length, v_width, v_height = get_video_properties(cap)
        cap.set(1, index)
        frame_i = 0
        for i in range(30):
            ret, frame = cap.read()
            frame_i += 1
            if not ret:
                break

            cv2.imshow('frame', frame)
            if cv2.waitKey(50) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
        cv2.waitKey(100)
        cap.release()
    cv2.destroyAllWindows()