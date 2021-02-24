import cv2
import numpy as np
import matplotlib.pyplot as plt


class CourtReference:
    def __init__(self):
        self.baseline_top = ((10, 10), (1107, 10))
        self.baseline_bottom = ((10, 2388), (1107, 2388))
        self.net = ((10, 1199), (1107, 1199))
        self.left_court_line = ((10, 10), (10, 2388))
        self.right_court_line = ((1107, 10), (1107, 2388))
        self.left_inner_line = ((147, 10), (147, 2388))
        self.right_inner_line = ((970, 10), (970, 2388))
        self.middle_line = ((558, 559), (558, 1839))
        self.top_inner_line = ((147, 559), (970, 559))
        self.bottom_inner_line = ((147, 1839), (970, 1839))
        self.line_width = 5
        self.court_width = 1127
        self.court_height = 2408
        self.court = cv2.cvtColor(cv2.imread('court_configurations/court_reference.png'), cv2.COLOR_BGR2GRAY)

    def build_court_reference(self):
        court = np.zeros((self.court_height, self.court_width), dtype=np.int)
        cv2.line(court, *self.baseline_top, 1, self.line_width)
        cv2.line(court, *self.baseline_bottom, 1, self.line_width)
        cv2.line(court, *self.net, 1, self.line_width)
        cv2.line(court, *self.top_inner_line, 1, self.line_width)
        cv2.line(court, *self.bottom_inner_line, 1, self.line_width)
        cv2.line(court, *self.left_court_line, 1, self.line_width)
        cv2.line(court, *self.right_court_line, 1, self.line_width)
        cv2.line(court, *self.left_inner_line, 1, self.line_width)
        cv2.line(court, *self.right_inner_line, 1, self.line_width)
        cv2.line(court, *self.middle_line, 1, self.line_width)
        plt.imsave('court_configurations/court_reference.png', court, cmap='gray')
        self.court = court
        return court

    def get_important_lines(self):
        lines = [*self.baseline_top, *self.baseline_bottom, *self.net, *self.left_court_line, *self.right_court_line,
                 *self.left_inner_line, *self.right_inner_line, *self.middle_line,
                 *self.top_inner_line, *self.bottom_inner_line]
        return lines
