import imutils

from src.court_detection import CourtDetector
from src.detection import DetectionModel
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
import cv2


class Statistics:
    def __init__(self, court_tracker: CourtDetector, players_detection: DetectionModel):
        self.court_tracker = court_tracker
        self.players_detection = players_detection

    def get_player_position_heatmap(self, pit_size=10):
        feet_bottom, feet_top = self.players_detection.calculate_feet_positions(self.court_tracker)
        court_width = self.court_tracker.court_reference.court_total_width
        court_height = self.court_tracker.court_reference.court_total_height
        heatmap = np.zeros((court_height // pit_size, court_width // pit_size))
        for x, y in feet_bottom:
            x = int(x)
            y = int(y)
            heatmap[y // pit_size, x // pit_size] += 1
        return heatmap

    def display_heatmap(self, heatmap, image):
        h, w = image.shape
        y, x = np.mgrid[0:h, 0:w]
        heatmap = imutils.resize(heatmap, w, h)

        heatmapshow = None
        heatmapshow = cv2.normalize(heatmap, heatmapshow, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        heatmapshow = cv2.applyColorMap(heatmapshow, cv2.COLORMAP_JET)
        cv2.imshow("Heatmap", heatmapshow)
        cv2.waitKey(50)

        mycmap = transparent_cmap(plt.cm.Reds)


        fig, ax = plt.subplots(1, 1)
        ax.imshow(image)
        cb = ax.contourf(x, y, heatmapshow, 15, cmap=mycmap)
        plt.colorbar(cb)
        plt.show()


def transparent_cmap(cmap, N=255):
    "Copy colormap and set alpha values"

    mycmap = cmap
    mycmap._init()
    mycmap._lut[:, -1] = np.linspace(0, 0.8, N + 4)
    return mycmap


if __name__ == "__main__":
    court = CourtDetector()
    ref = court.court_reference.court
    print(ref.shape)
