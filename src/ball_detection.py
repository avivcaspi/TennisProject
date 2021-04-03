from collections import deque
import numpy as np
import argparse
import imutils
import cv2
import torch
import torchvision
import torch.nn as nn
from PIL import Image, ImageDraw

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
    def __init__(self, save_state):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.detector = BallTrackerNet()
        saved_state_dict = torch.load(save_state)
        self.detector.load_state_dict(saved_state_dict['model_state'])
        self.detector.eval().to(self.device)

        self.current_frame = None
        self.last_frame = None
        self.before_last_frame = None

        self.movement_threshold = 200

        self.video_width = None
        self.video_height = None
        self.model_input_width = 640
        self.model_input_height = 360

        self.xy_coordinates = [(None, None), (None, None)]

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
            self.xy_coordinates.append((x, y))


if __name__ == "__main__":

    ball_detector = BallDetector('saved states/tracknet_weights_lr_0.05_epochs_280.pth')
    cap = cv2.VideoCapture('../videos/vid27.mp4')
    # get videos properties
    fps, length, v_width, v_height = get_video_properties(cap)
    output_video = cv2.VideoWriter('output/ball_detection.avi',
                                   cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (v_width, v_height))
    frame_i = 0
    while True:
        ret, frame = cap.read()
        frame_i += 1
        if not ret:
            break

        ball_detector.detect_ball(frame)
        q = ball_detector.xy_coordinates[-4:]
        PIL_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        PIL_image = Image.fromarray(PIL_image)
        for i in range(len(q)):
            if q[i][0] is not None:
                draw_x = q[i][0]
                draw_y = q[i][1]
                bbox = (draw_x - 2, draw_y - 2, draw_x + 2, draw_y + 2)
                draw = ImageDraw.Draw(PIL_image)
                draw.ellipse(bbox, outline='yellow')

            # Convert PIL image format back to opencv image format
            frame = cv2.cvtColor(np.array(PIL_image), cv2.COLOR_RGB2BGR)
        # write image to output_video
        output_video.write(frame)
        cv2.imshow('df', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    output_video.release()
    cv2.destroyAllWindows()
