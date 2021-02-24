import numpy as np
import cv2
from matplotlib import pyplot as plt
from sympy import Line
from itertools import combinations
from court_reference import CourtReference


class CourtDetector:
    def __init__(self):
        self.colour_threshold = 200
        self.dist_tau = 3
        self.intensity_threshold = 40
        self.court_reference = CourtReference()
        self.v_width = 0
        self.v_height = 0
        self.frame = None
        self.gray = None
        self.court_warp_matrix = None
        self.game_warp_matrix = None
        self.baseline_top = None
        self.baseline_bottom = None
        self.net = None
        self.left_court_line = None
        self.right_court_line = None
        self.left_inner_line = None
        self.right_inner_line = None
        self.middle_line = None
        self.top_inner_line = None
        self.bottom_inner_line = None

    def detect(self, frame):
        self.frame = frame
        self.v_height, self.v_width = frame.shape[:2]
        self.gray = self._threshold(frame)

        filtered = self._filter_pixels(self.gray)

        horizontal_lines, vertical_lines = self._detect_lines(filtered)

        self.court_warp_matrix, self.game_warp_matrix = self._find_homography(horizontal_lines, vertical_lines)

        self.find_lines_location()
        '''game_warped = cv2.warpPerspective(self.frame, self.game_warp_matrix,
                                          (self.court_reference.court.shape[1], self.court_reference.court.shape[0]))
        cv2.imwrite('../report/warped_game_1.png', game_warped)'''

    def _threshold(self, frame):
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
        lines = np.squeeze(lines)
        display_lines_on_frame(self.frame.copy(), [], lines)

        horizontal, vertical = self._classify_lines(lines)
        display_lines_on_frame(self.frame.copy(), horizontal, vertical)

        horizontal, vertical = self._merge_lines(horizontal, vertical)

        display_lines_on_frame(self.frame.copy(), horizontal, vertical)

        return horizontal, vertical

    def _classify_lines(self, lines):
        # TODO find better way to classify lines
        # TODO maybe set the 2 * dy to be dynamic
        horizontal = []
        vertical = []
        highest_vertical_y = np.inf
        lowest_vertical_y = 0
        for line in lines:
            x1, y1, x2, y2 = line
            dx = abs(x1 - x2)
            dy = abs(y1 - y2)
            if dx > 2 * dy:
                horizontal.append(line)
            else:
                vertical.append(line)
                highest_vertical_y = min(highest_vertical_y, y1, y2)
                lowest_vertical_y = max(lowest_vertical_y, y1, y2)
        clean_horizontal = []
        h = lowest_vertical_y - highest_vertical_y
        lowest_vertical_y += h / 15
        highest_vertical_y -= h * 2 / 15
        for line in horizontal:
            x1, y1, x2, y2 = line
            if lowest_vertical_y > y1 > highest_vertical_y and lowest_vertical_y > y1 > highest_vertical_y:
                clean_horizontal.append(line)
        return clean_horizontal, vertical

    def _classify_vertical(self, vertical, width):
        # TODO use vertical classification for improvement
        vertical_lines = []
        vertical_left = []
        vertical_right = []
        right_th = width * 4 / 7
        left_th = width * 3 / 7
        for line in vertical:
            x1, y1, x2, y2 = line
            if x1 < left_th or x2 < left_th:
                vertical_left.append(line)
            elif x1 > right_th or x2 > right_th:
                vertical_right.append(line)
            else:
                vertical_lines.append(line)
        return vertical_lines, vertical_left, vertical_right

    def _merge_lines(self, horizontal_lines, vertical_lines):
        horizontal_lines = sorted(horizontal_lines, key=lambda item: item[0])
        mask = [True] * len(horizontal_lines)
        new_horizontal_lines = []
        for i, line in enumerate(horizontal_lines):
            if mask[i]:
                for j, s_line in enumerate(horizontal_lines[i + 1:]):
                    if mask[i + j + 1]:
                        x1, y1, x2, y2 = line
                        x3, y3, x4, y4 = s_line
                        dy = abs(y3 - y2)
                        if dy < 10:
                            points = sorted([(x1, y1), (x2, y2), (x3, y3), (x4, y4)], key=lambda x: x[0])
                            line = np.array([*points[0], *points[-1]])
                            mask[i + j + 1] = False
                new_horizontal_lines.append(line)

        vertical_lines = sorted(vertical_lines, key=lambda item: item[1])
        xl, yl, xr, yr = (0, self.v_height * 6 / 7, self.v_width, self.v_height * 6 / 7)
        mask = [True] * len(vertical_lines)
        new_vertical_lines = []
        for i, line in enumerate(vertical_lines):
            if mask[i]:
                for j, s_line in enumerate(vertical_lines[i + 1:]):
                    if mask[i + j + 1]:
                        x1, y1, x2, y2 = line
                        x3, y3, x4, y4 = s_line
                        xi, yi = line_intersection(((x1, y1), (x2, y2)), ((xl, yl), (xr, yr)))
                        xj, yj = line_intersection(((x3, y3), (x4, y4)), ((xl, yl), (xr, yr)))

                        dx = abs(xi - xj)
                        if dx < 10:
                            points = sorted([(x1, y1), (x2, y2), (x3, y3), (x4, y4)], key=lambda x: x[1])
                            line = np.array([*points[0], *points[-1]])
                            mask[i + j + 1] = False

                new_vertical_lines.append(line)
        return new_horizontal_lines, new_vertical_lines

    def _find_homography(self, horizontal_lines, vertical_lines):
        # TODO maybe use random sampling of lines until threshold score reached
        max_score = -np.inf
        max_mat = None
        max_inv_mat = None
        k = 0
        for horizontal_pair in list(combinations(horizontal_lines, 2)):
            for vertical_pair in list(combinations(vertical_lines, 2)):
                h1, h2 = horizontal_pair
                v1, v2 = vertical_pair
                i1 = line_intersection((tuple(h1[:2]), tuple(h1[2:])), (tuple(v1[0:2]), tuple(v1[2:])))
                i2 = line_intersection((tuple(h1[:2]), tuple(h1[2:])), (tuple(v2[0:2]), tuple(v2[2:])))
                i3 = line_intersection((tuple(h2[:2]), tuple(h2[2:])), (tuple(v1[0:2]), tuple(v1[2:])))
                i4 = line_intersection((tuple(h2[:2]), tuple(h2[2:])), (tuple(v2[0:2]), tuple(v2[2:])))

                intersections = [i1, i2, i3, i4]
                intersections = sort_intersection_points(intersections)

                for i, configuration in self.court_reference.court_conf.items():
                    matrix, _ = cv2.findHomography(np.float32(configuration), np.float32(intersections), method=0)
                    inv_matrix = cv2.invert(matrix)[1]
                    confi_score = self._get_confi_score(matrix)

                    if max_score < confi_score:
                        max_score = confi_score
                        max_mat = matrix
                        max_inv_mat = inv_matrix

                    k += 1

        frame = self.frame.copy()
        court = self.add_court_overlay(frame, max_mat, (255, 0, 0))
        print(f'Score = {max_score}')
        cv2.imshow('court', court)
        if cv2.waitKey(0) & 0xff == 27:
            cv2.destroyAllWindows()
        print(k)

        return max_mat, max_inv_mat

    def _get_confi_score(self, matrix):
        court = cv2.warpPerspective(self.court_reference.court, matrix, self.frame.shape[1::-1])
        court[court > 0] = 1
        gray = self.gray.copy()
        gray[gray > 0] = 1
        correct = court * gray
        wrong = court - correct
        c_p = np.sum(correct)
        w_p = np.sum(wrong)
        return c_p - 0.5 * w_p

    def add_court_overlay(self, frame, homography, overlay_color=(255, 255, 255)):
        court = cv2.warpPerspective(self.court_reference.court, homography, frame.shape[1::-1])
        frame[court == 255, :] = overlay_color
        return frame

    def find_lines_location(self):
        p = np.array(self.court_reference.get_important_lines(), dtype=np.float32).reshape((-1, 1, 2))
        lines = cv2.perspectiveTransform(p, self.court_warp_matrix).reshape(-1)
        self.baseline_top = lines[:4]
        self.baseline_bottom = lines[4:8]
        self.net = lines[8:12]
        self.left_court_line = lines[12:16]
        self.right_court_line = lines[16:20]
        self.left_inner_line = lines[20:24]
        self.right_inner_line = lines[24:28]
        self.middle_line = lines[28:32]
        self.top_inner_line = lines[32:36]
        self.bottom_inner_line = lines[36:40]
        display_lines_on_frame(self.frame.copy(), [self.baseline_top, self.baseline_bottom, self.net, self.top_inner_line, self.bottom_inner_line],
                               [self.left_court_line, self.right_court_line, self.right_inner_line, self.left_inner_line, self.middle_line])


def sort_intersection_points(intersections):
    y_sorted = sorted(intersections, key=lambda x: x[1])
    p12 = y_sorted[:2]
    p34 = y_sorted[2:]
    p12 = sorted(p12, key=lambda x: x[0])
    p34 = sorted(p34, key=lambda x: x[0])
    return p12 + p34


def line_intersection(line1, line2):
    l1 = Line(line1[0], line1[1])
    l2 = Line(line2[0], line2[1])

    intersection = l1.intersection(l2)
    return intersection[0].coordinates


def display_lines_on_frame(frame, horizontal=(), vertical=()):
    '''cv2.line(frame, (int(len(frame[0]) * 4 / 7), 0), (int(len(frame[0]) * 4 / 7), 719), (255, 255, 0), 2)
    cv2.line(frame, (int(len(frame[0]) * 3 / 7), 0), (int(len(frame[0]) * 3 / 7), 719), (255, 255, 0), 2)'''
    for line in horizontal:
        x1, y1, x2, y2 = line
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(frame, (x1, y1), 1, (255, 0, 0), 2)
        cv2.circle(frame, (x2, y2), 1, (255, 0, 0), 2)

    for line in vertical:
        x1, y1, x2, y2 = line
        cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.circle(frame, (x1, y1), 1, (255, 0, 0), 2)
        cv2.circle(frame, (x2, y2), 1, (255, 0, 0), 2)

    cv2.imshow('court', frame)
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()
    # cv2.imwrite('../report/t.png', frame)
    return frame


def display_lines_and_points_on_frame(frame, lines=(), points=(), line_color=(0, 0, 255), point_color=(255, 0, 0)):
    for line in lines:
        x1, y1, x2, y2 = line
        frame = cv2.line(frame, (x1, y1), (x2, y2), line_color, 2)
    for p in points:
        frame = cv2.circle(frame, p, 2, point_color, 2)

    cv2.imshow('court', frame)
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()
    return frame


if __name__ == '__main__':
    filename = '../images/img2.jpg'
    img = cv2.imread(filename)
    import time

    s = time.time()
    court_detector = CourtDetector()
    court_detector.detect(img)
    print(f'time = {time.time() - s}')
