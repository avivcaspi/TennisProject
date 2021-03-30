from collections import deque
import numpy as np
import argparse
import imutils
import cv2
import torchvision
import torch.nn as nn

from src.court_detection import CourtDetector


class BallDetector:
    def __init__(self):
        self.current_frame = None
        self.last_frame = None
        self.next_frame = None
        self.movement_threshold = 200

    def detect_ball(self, next_frame, court):
        gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
        self.last_frame = self.current_frame
        self.current_frame = self.next_frame
        self.next_frame = gray.copy()
        if self.last_frame is not None:
            first_motion = abs(self.current_frame - self.last_frame)
            first_motion = cv2.threshold(first_motion, self.movement_threshold, 255, cv2.THRESH_BINARY)[1]
            second_motion = abs(self.next_frame - self.last_frame)
            second_motion = cv2.threshold(second_motion, self.movement_threshold, 255, cv2.THRESH_BINARY)[1]
            motion_matrix = first_motion.copy()
            motion_matrix[second_motion > 0] = 0
            court = cv2.dilate(court, cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20)))
            motion_matrix[court > 0] = 0
            #motion_matrix = cv2.morphologyEx(motion_matrix, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10)))
            motion_matrix = cv2.dilate(motion_matrix, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20)))

            contours, _ = cv2.findContours(motion_matrix, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

            contours_poly = [cv2.approxPolyDP(c, 3, True) for c in contours if 50 < cv2.contourArea(c) < 100]
            f = cv2.cvtColor(motion_matrix, cv2.COLOR_GRAY2BGR)
            z = np.zeros_like(f)
            for i, c in enumerate(contours):
                if 200 < cv2.contourArea(c) < 1500:
                    x, y, w, h = cv2.boundingRect(c)
                    cv2.rectangle(f, (x, y), (x + w, y + h), (0, 255, 0), 2)
                #cv2.drawContours(z, contours_poly, i, (255, 0, 0))
            cv2.imshow('sdf',self.current_frame * (motion_matrix // 255))
            if cv2.waitKey(100) & 0xff == 27:
                cv2.destroyAllWindows()


if __name__ == "__main__":
    court_detector = CourtDetector()
    ball_detector = BallDetector()
    cap = cv2.VideoCapture('../videos/vid1.mp4')

    frame_i = 0
    while True:
        ret, frame = cap.read()
        frame_i += 1
        if not ret:
            break

        if frame_i == 1:
            court_detector.detect(frame)
        court_detector.track_court(frame)

        white_ref = court_detector.court_reference.get_court_mask()
        white_mask = cv2.warpPerspective(white_ref, court_detector.court_warp_matrix[-1], frame.shape[1::-1])
        image_court = frame.copy()
        image_court[white_mask == 0, :] = (0, 0, 0)
        ball_detector.detect_ball(image_court, court_detector.get_warped_court())
    cap.release()
    cv2.destroyAllWindows()
