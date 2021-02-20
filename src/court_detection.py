import numpy as np
import cv2
from matplotlib import pyplot as plt
from sympy import Point, Line


def build_court_reference():
    court = np.zeros((2408,1127), dtype=np.int)
    cv2.line(court, (10, 10), (1107, 10), 1, 2)
    cv2.line(court, (10, 2388), (1107, 2388), 1, 2)
    cv2.line(court, (10, 1199), (1107, 1199), 1, 2)
    cv2.line(court, (147, 559), (970, 559), 1, 2)
    cv2.line(court, (147, 1839), (970, 1839), 1, 2)
    cv2.line(court, (10, 10), (10, 2388), 1, 2)
    cv2.line(court, (1107, 10), (1107, 2388), 1, 2)
    cv2.line(court, (147, 10), (147, 2388), 1, 2)
    cv2.line(court, (970, 10), (970, 2388), 1, 2)
    cv2.line(court, (558, 559), (558, 1839), 1, 2)
    plt.imsave('court_reference.png', court, cmap='gray')


class CourtDetector:
    def __init__(self):
        self.colour_threshold = 200
        self.dist_tau = 3
        self.intensity_threshold = 40
        self.court_conf = {1: [(10, 10), (1107, 10), (10, 2388), (1107, 2388)],
                           2: [(147, 10), (970, 10), (147, 2388), (970, 2388)],
                           3: [(147, 10), (1107, 10), (147, 2388), (1107, 2388)],
                           4: [(10, 10), (970, 10), (10, 2388), (970, 2388)],
                           5: [(147, 559), (970, 559), (147, 1839), (970, 1839)],
                           6: [(147, 559), (970, 559), (147, 2388), (970, 2388)],
                           7: [(147, 10), (970, 10), (147, 1839), (970, 1839)],
                           8: [(970, 10), (1107, 10), (970, 2388), (1107, 2388)],
                           9: [(10, 10), (147, 10), (10, 2388), (147, 2388)],
                           10: [(147, 559), (558, 559), (147, 1839), (558, 1839)],
                           11: [(558, 559), (970, 559), (558, 1839), (970, 1839)]}

    def detect(self, frame):
        gray = self._threshold(frame)

        gray = self._filter_pixels(gray)

        horizontal_lines, vertical_lines = self._detect_lines(gray)

        self._find_homography(horizontal_lines, vertical_lines)

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
        frame = self.frame.copy()
        cv2.line(frame, (int(len(gray[0]) * 4 / 7), 0), (int(len(gray[0]) * 4 / 7), 719), (255,255,0),2)
        cv2.line(frame, (int(len(gray[0]) * 3 / 7), 0), (int(len(gray[0]) * 3 / 7), 719), (255, 255, 0), 2)
        for line in horizontal:
            for x1, y1, x2, y2 in line:
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(frame, (x1, y1), 1, (255, 0, 0), 2)
                cv2.circle(frame, (x2, y2), 1, (255, 0, 0), 2)

        for line in vertical:
            for x1, y1, x2, y2 in line:
                cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.circle(frame, (x1, y1), 1, (255, 0, 0), 2)
                cv2.circle(frame, (x2, y2), 1, (255, 0, 0), 2)

        cv2.imshow('court', frame)
        if cv2.waitKey(0) & 0xff == 27:
            cv2.destroyAllWindows()
        cv2.imwrite('houghlines5.jpg', frame)
        return horizontal, vertical

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

    def _classify_vertical(self, vertical, width):
        vertical_lines = []
        vertical_left = []
        vertical_right = []
        right_th = width * 4 / 7
        left_th = width * 3 / 7
        for line in vertical:
            for x1, y1, x2, y2 in line:
                if x1 < left_th or x2 < left_th:
                    vertical_left.append(line)
                elif x1 > right_th or x2 > right_th:
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

    def _find_homography(self, horizontal_lines, vertical_lines):
        h1 = horizontal_lines[0][0]
        h2 = horizontal_lines[2][0]
        v1 = vertical_lines[0][0]
        v2 = vertical_lines[4][0]
        frame = self.frame.copy()
        cv2.line(frame, (h1[ 0], h1[1]), (h1[ 2], h1[ 3]), (0, 255, 0), 2)
        cv2.line(frame, (h2[ 0], h2[ 1]), (h2[ 2], h2[3]), (0, 255, 0), 2)
        cv2.line(frame, (v1[ 0], v1[ 1]), (v1[ 2], v1[ 3]) ,(0, 255, 0), 2)
        cv2.line(frame, (v2[ 0], v2[ 1]), (v2[ 2], v2[ 3]), (0, 255, 0), 2)
        i1 = line_intersection((tuple(h1[:2]), tuple(h1[2:])), (tuple(v1[0:2]), tuple(v1[2:])))
        i2 = line_intersection((tuple(h1[:2]), tuple(h1[2:])), (tuple(v2[0:2]), tuple(v2[2:])))
        i3 = line_intersection((tuple(h2[:2]), tuple(h2[2:])), (tuple(v1[0:2]), tuple(v1[2:])))
        i4 = line_intersection((tuple(h2[:2]), tuple(h2[2:])), (tuple(v2[0:2]), tuple(v2[2:])))

        intersections = [i1, i2, i3, i4]
        intersections = sorted(intersections, key=lambda x: x[0])
        p1, p2 = sorted(intersections[:2], key=lambda x: x[1])
        p3, p4 = sorted(intersections[2:], key=lambda x: x[1])
        a = p2
        p2 = p3
        p3 = a
        cv2.circle(frame, p1, 2, (255, 0, 0), 2)
        cv2.circle(frame, p2, 2, (0, 255, 0), 2)
        cv2.circle(frame, p3, 2, (0, 0, 255), 2)
        cv2.circle(frame, p4, 2, (255, 255, 0), 2)
        cv2.imshow('court', frame)
        if cv2.waitKey(0) & 0xff == 27:
            cv2.destroyAllWindows()



def line_intersection(line1, line2):

    l1 = Line(line1[0], line1[1])
    l2 = Line(line2[0], line2[1])

    intersection = l1.intersection(l2)
    return intersection[0].coordinates


court_reference = cv2.cvtColor(cv2.imread('court_reference.png'), cv2.COLOR_BGR2GRAY)
filename = '../images/img1.jpg'
img = cv2.imread(filename)
import time
s = time.time()
court_detector = CourtDetector()
court_detector.detect(img)
print(f'time = {time.time() - s}')
