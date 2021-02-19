import numpy as np
import cv2
from matplotlib import pyplot as plt
from sympy import Point, Line


class CourtDetector:
    def __init__(self):
        self.colour_threshold = 200
        self.dist_tau = 3
        self.intensity_threshold = 40

    def detect(self, frame):
        gray = self._threshold(frame)
        cv2.imwrite('before_filtering.png', gray)

        gray = self._filter_pixels(gray)
        cv2.imwrite('after_filtering.png', gray)

        self._detect_lines(gray)


    def _threshold(self, frame):
        self.frame = frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray[gray > self.colour_threshold] = 255
        gray[gray < 255] = 0
        return gray

    def _filter_pixels(self, gray):
        # TODO change this to something more efficient
        for i in range(self.dist_tau, len(gray) - self.dist_tau):
            for j in range(self.dist_tau, len(gray[0]) - self.dist_tau):
                if gray[i, j] == 0:
                    continue
                if (gray[i, j] - gray[i + self.dist_tau, j] > self.intensity_threshold and
                        gray[i, j] - gray[i - self.dist_tau, j] > self.intensity_threshold):
                    continue

                if (gray[i, j] - gray[i, j + self.dist_tau] > self.intensity_threshold and
                        gray[i, j] - gray[i, j - self.dist_tau] > self.intensity_threshold):
                    continue
                gray[i, j] = 0
        return gray

    def _detect_lines(self, gray):
        minLineLength = 100
        maxLineGap = 20
        lines = cv2.HoughLinesP(gray, 1, np.pi / 180, 80, minLineLength=minLineLength, maxLineGap=maxLineGap)
        horizontal, vertical = self._classify_lines(lines)

        horizontal, vertical = self._merge_lines(horizontal, vertical)

        cv2.line(self.frame, (int(len(gray[0]) * 4 / 7), 0), (int(len(gray[0]) * 4 / 7), 719), (255,255,0),2)
        cv2.line(self.frame, (int(len(gray[0]) * 3 / 7), 0), (int(len(gray[0]) * 3 / 7), 719), (255, 255, 0), 2)
        for line in horizontal:
            for x1, y1, x2, y2 in line:
                cv2.line(self.frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(self.frame, (x1, y1), 1, (255, 0, 0), 2)
                cv2.circle(self.frame, (x2, y2), 1, (255, 0, 0), 2)

        for line in vertical:
            for x1, y1, x2, y2 in line:
                cv2.line(self.frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.circle(self.frame, (x1, y1), 1, (255, 0, 0), 2)
                cv2.circle(self.frame, (x2, y2), 1, (255, 0, 0), 2)

        cv2.imshow('court', self.frame)
        if cv2.waitKey(0) & 0xff == 27:
            cv2.destroyAllWindows()
        cv2.imwrite('houghlines5.jpg', self.frame)
        return gray

    def _classify_lines(self, lines):
        horizontal = []
        vertical = []

        for line in lines:
            for x1, y1, x2, y2 in line:
                dx = abs(x1 - x2)
                dy = abs(y1 - y2)
                if dx > dy:
                    horizontal.append(line)
                else:
                    vertical.append(line)
        return horizontal, vertical

    def _classify_vertical(self, vertical_lines, width):
        vertical_lines = []
        vertical_left = []
        vertical_right = []
        right_th = width * 4 / 7
        left_th = width * 3 / 7
        for line in vertical_lines:
            for x1, y1, x2, y2 in line:
                if x1 < left_th or x2 < left_th:
                    vertical_left.append(line)
                elif x1 > right_th or x2 >right_th:
                    vertical_right.append(line)
                else:
                    vertical_lines.append(line)
        return vertical_lines, vertical_left, vertical_right

    def _merge_lines(self, horizontal_lines, vertical_lines):
        horizontal_lines = sorted(horizontal_lines, key=lambda item: item[0, 0])
        mask = [True]*len(horizontal_lines)
        new_horizontal_lines = []
        for i, line in enumerate(horizontal_lines):
            if mask[i]:
                for j, s_line in enumerate(horizontal_lines[i+1:]):
                    if mask[i+j+1]:
                        x1, y1, x2, y2 = line[0]
                        x3, y3, x4, y4 = s_line[0]
                        dy = abs(y3 - y2)
                        if dy < 10:
                            line[0,2] = x4
                            line[0,3] = y4
                            mask[i+j+1] = False
                new_horizontal_lines.append(line)

        vertical_lines = sorted(vertical_lines, key=lambda item: item[0,1])
        #reference_horizontal_line = max(new_horizontal_lines, key=lambda item: item[0, 1])
        xl,yl, xr,yr = (100, 300, 1000, 300)
        mask = [True] * len(vertical_lines)
        new_vertical_lines = []
        for i, line in enumerate(vertical_lines):
            if mask[i]:
                for j, s_line in enumerate(vertical_lines[i+1:]):
                    if mask[i+j+1]:
                        x1, y1, x2, y2 = line[0]
                        x3, y3, x4, y4 = s_line[0]
                        xi, yi = line_intersection(((x1,y1), (x2,y2)), ((xl,yl), (xr,yr)))
                        xj, yj = line_intersection(((x3, y3), (x4, y4)), ((xl, yl), (xr, yr)))
                        dx = abs(xi - xj)
                        if dx < 15:
                            line[0,2] = x4
                            line[0,3] = y4
                            mask[i+j+1] = False
                new_vertical_lines.append(line)
        return new_horizontal_lines, new_vertical_lines


def line_intersection(line1, line2):

    l1 = Line(line1[0], line1[1])
    l2 = Line(line2[0], line2[1])

    intersection = l1.intersection(l2)
    return intersection[0].coordinates


filename = '../images/img1.jpg'
img = cv2.imread(filename)
import time
s = time.time()
court_detector = CourtDetector()
court_detector.detect(img)
print(f'time = {time.time() - s}')
