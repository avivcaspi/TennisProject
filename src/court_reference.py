import cv2
import numpy as np
import matplotlib.pyplot as plt


class CourtReference:
    def __init__(self):
        # TODO change location with regards to line width (all line needs to be inside court)
        self.baseline_top = ((284, 559), (1391, 559))
        self.baseline_bottom = ((284, 2937), (1391, 2937))
        self.net = ((284, 1748), (1391, 1748))
        self.left_court_line = ((284, 559), (284, 2937))
        self.right_court_line = ((1391, 559), (1391, 2937))
        self.left_inner_line = ((421, 559), (421, 2937))
        self.right_inner_line = ((1244, 559), (1244, 2937))
        self.middle_line = ((832, 1108), (832, 2388))
        self.top_inner_line = ((421, 1108), (1244, 1108))
        self.bottom_inner_line = ((421, 2388), (1244, 2388))

        self.court_conf = {1: [*self.baseline_top, *self.baseline_bottom],
                           2: [self.left_inner_line[0], self.right_inner_line[0], self.left_inner_line[1],
                               self.right_inner_line[1]],
                           3: [self.left_inner_line[0], self.right_court_line[0], self.left_inner_line[1],
                               self.right_court_line[1]],
                           4: [self.left_court_line[0], self.right_inner_line[0], self.left_court_line[1],
                               self.right_inner_line[1]],
                           5: [*self.top_inner_line, *self.bottom_inner_line],
                           6: [*self.top_inner_line, self.left_inner_line[1], self.right_inner_line[1]],
                           7: [self.left_inner_line[0], self.right_inner_line[0], *self.bottom_inner_line],
                           8: [self.right_inner_line[0], self.right_court_line[0], self.right_inner_line[1],
                               self.right_court_line[1]],
                           9: [self.left_court_line[0], self.left_inner_line[0], self.left_court_line[1],
                               self.left_inner_line[1]],
                           10: [self.top_inner_line[0], self.middle_line[0], self.bottom_inner_line[0],
                                self.middle_line[1]],
                           11: [self.middle_line[0], self.top_inner_line[1], self.middle_line[1],
                                self.bottom_inner_line[1]],
                           12: [*self.bottom_inner_line, self.left_inner_line[1], self.right_inner_line[1]]}
        self.line_width = 5
        self.court_width = 1117
        self.court_height = 2408
        self.top_bottom_border = 549
        self.right_left_border = 274
        self.court = cv2.cvtColor(cv2.imread('court_configurations/court_reference.png'), cv2.COLOR_BGR2GRAY)

    def build_court_reference(self):
        court = np.zeros((self.court_height + 2 * self.top_bottom_border, self.court_width + 2 * self.right_left_border), dtype=np.int)
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

    def save_all_court_configurations(self):
        for i, conf in self.court_conf.items():
            c = cv2.cvtColor(255 - self.court, cv2.COLOR_GRAY2BGR)
            for p in conf:
                c = cv2.circle(c, p, 15, (0, 0, 255), 30)
            cv2.imwrite(f'court_configurations/court_conf_{i}.png', c)


if __name__ == '__main__':
    c = CourtReference()
    c.build_court_reference()