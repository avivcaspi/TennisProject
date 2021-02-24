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
        self.RACKET_LABEL = 430
        self.BALL_LABEL = 370
        self.PERSON_SCORE_MIN = 0.4
        self.RACKET_SCORE_MIN = 0.6
        self.BALL_SCORE_MIN = 0.6
        self.im_diff = ImageDiff()
        self.backSub = cv2.createBackgroundSubtractorKNN()

    def detect_objects(self, image):
        mask = self.find_canadicate(image)
        image[mask == 0, :] = (0,0,0)
        boxes = np.zeros_like(image)

        # creating torch.tensor from the image ndarray
        frame_t = image.transpose((2, 0, 1)) / 255
        frame_tensor = torch.from_numpy(frame_t).unsqueeze(0).type(self.dtype)

        # Finding boxes and keypoints
        with torch.no_grad():
            # forward pass
            p = self.detection_model(frame_tensor)

        for box, label, score in zip(p[0]['boxes'][:], p[0]['labels'], p[0]['scores']):
            if label == self.PERSON_LABEL and score > self.PERSON_SCORE_MIN:
                cv2.rectangle(boxes, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), [255, 0, 255], 2)
                cv2.putText(boxes, 'Person %.3f' % score, (int(box[0]) - 10, int(box[1] - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
            if label == self.RACKET_LABEL and score > self.RACKET_SCORE_MIN:
                cv2.rectangle(boxes, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), [255, 0, 0], 2)
                cv2.putText(image, 'Racket %.3f' % score, (int(box[0]) - 10, int(box[1] - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            if label == self.BALL_LABEL and score > self.BALL_SCORE_MIN:
                cv2.rectangle(boxes, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), [0, 255, 0], 2)
                cv2.putText(image, 'Ball  %.3f' % score, (int(box[0]) - 10, int(box[1] - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return boxes

    def find_canadicate(self, image):
        frame = image.copy()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.medianBlur(frame, 3)

        fgMask = self.backSub.apply(frame)
        fgMask = cv2.threshold(fgMask, 10, 1, cv2.THRESH_BINARY)[1]

        diff = self.im_diff.diff(frame)

        res = diff * fgMask * 255
        res = cv2.dilate(res, np.ones((25,25)))
        contours, _ = cv2.findContours(res, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours_poly = None
        boundRect = None
        max_area = 0
        max_c = None
        for c in contours:
            if cv2.contourArea(c) > max_area:
                max_area = cv2.contourArea(c)
                max_c = c

                contours_poly = cv2.approxPolyDP(c, 3, True)
                boundRect = cv2.boundingRect(contours_poly)

        drawing = np.zeros((res.shape[0], res.shape[1], 3), dtype=np.uint8)


        color = (0,0, 255)
        #cv2.drawContours(drawing, contours_poly, i, color)
        cv2.rectangle(drawing, (int(boundRect[0]), int(boundRect[1])),(int(boundRect[0] + boundRect[2]), int(boundRect[1] + boundRect[3])), color, 2,)

        '''cv2.imshow('Contours', res)
        c = image.copy()
        c[res == 0] = 0
        #cv2.imshow('res',  c)

        if cv2.waitKey(0) & 0xff == 27:
            cv2.destroyAllWindows()'''
        return res


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
