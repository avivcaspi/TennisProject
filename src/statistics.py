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

    def get_player_position_heatmap(self, pit_size=80):
        feet_bottom, feet_top = self.players_detection.calculate_feet_positions(self.court_tracker)
        court_width = self.court_tracker.court_reference.court_total_width
        court_height = self.court_tracker.court_reference.court_total_height

        heatmap = np.zeros((court_height // pit_size, court_width // pit_size))
        for x, y in feet_bottom:
            x = int(x)
            y = int(y)
            heatmap[y // pit_size, x // pit_size] += 1
        for x, y in feet_top:
            x = int(x)
            y = int(y)
            heatmap[y // pit_size, x // pit_size] += 1

        return heatmap

    def display_heatmap(self, heatmap, image=None):
        if image is not None:
            h, w = image.shape

            heatmap = imutils.resize(heatmap, w, h)
            heatmap = heatmap[:h, :w]
            image = imutils.resize(image, 500)

        heatmap = imutils.resize(heatmap, 500)

        fig = plt.figure(figsize=(5,10))
        # Define the canvas as 1*1 division, and draw on the first position
        ax = fig.add_subplot()

        # Draw and select the color fill style of the heat map, select hot here

        if image is not None:
            im2 = ax.imshow(image, cmap='gray')
            pass
        im = ax.imshow(heatmap, alpha=0.5, cmap=plt.cm.bwr)

        # Add the color scale bar on the right
        # plt.colorbar(im)
        # Add title

        # show
        plt.show()


if __name__ == "__main__":
    court = CourtDetector()
    ref = court.court_reference.court
    heatmap = np.zeros((350, 166))
    heatmap[30:50, 10:20] = 30
    heatmap[100:150, 40:90] = 20
    heatmap[200:250, 40:70] = 10
    stats = Statistics(court, None)
    stats.display_heatmap(heatmap, ref)
    print(ref.shape)
