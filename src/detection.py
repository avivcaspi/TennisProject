import os
import cv2
import torch
import torchvision
import numpy as np
import pandas as pd


class DetectionModel:
    def __init__(self, dtype=torch.FloatTensor):
        self.detection_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        self.detection_model.type(dtype)  # Also moves model to GPU if available
        self.detection_model.eval()
        self.dtype = dtype
        self.PERSON_LABEL = 1
        self.RACKET_LABEL = 43
        self.BALL_LABEL = 37
        self.PERSON_SCORE_MIN = 0.85
        self.RACKET_SCORE_MIN = 0.6
        self.BALL_SCORE_MIN = 0.6
        self.v_width = 0
        self.v_height = 0
        self.player_1_boxes = []
        self.player_2_boxes = []
        self.persons_boxes = []
        self.counter = 0
        self.im_diff = ImageDiff()
        self.backSub = cv2.createBackgroundSubtractorKNN()

    def detect_player_1(self, image, court_detector):
        boxes = np.zeros_like(image)

        self.v_height, self.v_width = image.shape[:2]
        if len(self.player_1_boxes) == 0:
            court_type = 1
            white_ref = court_detector.court_reference.get_court_mask(court_type)
            white_mask = cv2.warpPerspective(white_ref, court_detector.court_warp_matrix, image.shape[1::-1])
            # TODO find different way to add more space at the top
            if court_type == 2:
                white_mask = cv2.dilate(white_mask, np.ones((50, 1)), anchor=(0, 0))
            image_court = image.copy()
            image_court[white_mask == 0, :] = (0, 0, 0)
            '''max_values = np.max(np.max(image_court, axis=1), axis=1)
            max_values_index = np.where(max_values > 0)[0]
            top_y = max_values_index[0]
            bottom_y = max_values_index[-1]'''
            # cv2.imwrite('../report/frame_only_court.png', image_court)
            '''cv2.imshow('res', image_court)
            if cv2.waitKey(0) & 0xff == 27:
                cv2.destroyAllWindows()'''

            # mask = self.find_canadicate(image)
            # image[mask == 0, :] = (0,0,0)
            # image_court = image_court[top_y:bottom_y, :, :]

            persons_boxes = self._detect(image_court)
            if len(persons_boxes) > 0:
                # biggest_box = sorted(persons_boxes, key=lambda x: area_of_box(x), reverse=True)[0]
                bottom_box = max(persons_boxes, key=lambda x: x[3])
                self.player_1_boxes.append(bottom_box)
        else:
            xt, yt, xb, yb = self.player_1_boxes[-1]
            xt, yt, xb, yb = int(xt), int(yt), int(xb), int(yb)
            margin = 100
            box_corners = (max(xt - margin, 0), max(yt - margin, 0), min(xb + margin, self.v_width), min(yb + margin, self.v_height))
            trimmed_image = image[max(yt - margin, 0): min(yb + margin, self.v_height), max(xt - margin, 0): min(xb + margin, self.v_width), :]
            '''cv2.imshow('res', trimmed_image)
            if cv2.waitKey(0) & 0xff == 27:
                cv2.destroyAllWindows()'''

            persons_boxes = self._detect(trimmed_image)
            if len(persons_boxes) > 0:
                c1 = center_of_box(self.player_1_boxes[-1])
                closest_box = None
                smallest_dist = np.inf
                for box in persons_boxes:
                    orig_box_location = (box_corners[0] + box[0], box_corners[1] + box[1], box_corners[0] + box[2], box_corners[1] + box[3])
                    c2 = center_of_box(orig_box_location)
                    distance = np.linalg.norm(np.array(c1) - np.array(c2))
                    if distance < smallest_dist:
                        smallest_dist = distance
                        closest_box = orig_box_location
                if smallest_dist < 100:
                    self.counter = 0
                    self.player_1_boxes.append(closest_box)
                else:
                    self.counter += 1
                    self.player_1_boxes.append(self.player_1_boxes[-1])
            else:
                self.player_1_boxes.append(self.player_1_boxes[-1])
        cv2.rectangle(boxes, (int(self.player_1_boxes[-1][0]), int(self.player_1_boxes[-1][1])),
                      (int(self.player_1_boxes[-1][2]), int(self.player_1_boxes[-1][3])), [255, 0, 255], 2)

        return boxes

    def _detect(self, image):
        # creating torch.tensor from the image ndarray
        frame_t = image.transpose((2, 0, 1)) / 255
        frame_tensor = torch.from_numpy(frame_t).unsqueeze(0).type(self.dtype)

        # Finding boxes and keypoints
        with torch.no_grad():
            # forward pass
            p = self.detection_model(frame_tensor)

        persons_boxes = []
        for box, label, score in zip(p[0]['boxes'][:], p[0]['labels'], p[0]['scores']):
            if label == self.PERSON_LABEL and score > self.PERSON_SCORE_MIN:
                '''cv2.rectangle(boxes, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), [255, 0, 255], 2)
                cv2.putText(boxes, 'Person %.3f' % score, (int(box[0]) - 10, int(box[1] - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)'''
                persons_boxes.append(box.detach().cpu())
        return persons_boxes

    def find_canadicate(self, image):
        frame = image.copy()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.medianBlur(frame, 3)

        fgMask = self.backSub.apply(frame)
        fgMask = cv2.threshold(fgMask, 10, 1, cv2.THRESH_BINARY)[1]

        diff = self.im_diff.diff(frame)

        res = diff * fgMask * 255
        res = cv2.dilate(res, np.ones((40, 25)))
        contours, _ = cv2.findContours(res, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours_poly = []
        boundRects = []
        max_area = 0
        max_c = None
        for c in contours:
            if 200 < cv2.contourArea(c) < 12000:
                contours_poly.append(cv2.approxPolyDP(c, 3, True))
                boundRects.append(cv2.boundingRect(contours_poly[-1]))

        drawing = np.zeros((res.shape[0], res.shape[1], 3), dtype=np.uint8)
        f = image.copy()
        mask = np.ones_like(image)
        for i, boundRect in enumerate(boundRects):
            mask[int(boundRect[1]):int(boundRect[1] + boundRect[3]), int(boundRect[0]):int(boundRect[0] + boundRect[2]), :] = (0,0,0)
            f = f * mask
            '''cv2.imshow('res', box)
            if cv2.waitKey(0) & 0xff == 27:
                cv2.destroyAllWindows()'''

            color = (0, 0, 255)
            cv2.drawContours(drawing, contours_poly, i, (255,0,0))
            cv2.rectangle(drawing, (int(boundRect[0]), int(boundRect[1])),
                          (int(boundRect[0] + boundRect[2]), int(boundRect[1] + boundRect[3])), color, 2)

        res = cv2.cvtColor(res, cv2.COLOR_GRAY2BGR)
        side_by_side = np.concatenate([res, image], axis=1)
        side_by_side = cv2.resize(side_by_side, (1920, 540))
        cv2.imshow('Contours', side_by_side)
        c = image.copy()
        c[res == 0] = 0

        if cv2.waitKey(0) & 0xff == 27:
            cv2.destroyAllWindows()
        return res


def center_of_box(box):
    height = box[3] - box[1]
    width = box[2] - box[0]
    return box[0] + width / 2, box[1] + height / 2


def area_of_box(box):
    height = box[3] - box[1]
    width = box[2] - box[0]
    return height * width


class ImageDiff:
    def __init__(self):
        self.last_image = None
        self.diff_image = None

    def diff(self, image):
        if self.last_image is None:
            self.last_image = image.copy()
            return np.ones_like(image)
        else:
            self.diff_image = abs(self.last_image - image)
            self.diff_image = cv2.threshold(self.diff_image, 200, 1, cv2.THRESH_BINARY)[1]
            return self.diff_image


if __name__ == "__main__":
    video = cv2.VideoCapture('../videos/vid3.mp4')
    model = DetectionModel()
    while True:
        ret, frame = video.read()

        if ret:
            model.find_canadicate(frame)

        else:
            break
