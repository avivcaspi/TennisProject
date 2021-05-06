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
        self.feet_bottom, self.feet_top = self.players_detection.calculate_feet_positions(self.court_tracker)
        self.top_dists_array = None
        self.bottom_dists_array = None

    def get_player_position_heatmap(self, pit_size=80):
        """
        Calculate the heatmap of both players positions in the video
        """
        court_width = self.court_tracker.court_reference.court_total_width
        court_height = self.court_tracker.court_reference.court_total_height
        heatmap_shape = (court_height // pit_size, court_width // pit_size)
        heatmap = np.zeros(heatmap_shape)
        for x, y in self.feet_bottom:
            x = int(x)
            y = int(y)
            heatmap[min(heatmap_shape[0] - 1,y // pit_size), min(heatmap_shape[1] - 1,x // pit_size)] += 1
        for x, y in self.feet_top:
            x = int(x)
            y = int(y)
            heatmap[min(heatmap_shape[0] - 1,y // pit_size), min(heatmap_shape[1] - 1,x // pit_size)] += 1

        return heatmap

    def display_heatmap(self, heatmap, image=None, cmap=plt.cm.bwr, title=''):
        """
        Display the heatmap on top of an image
        """

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
        im = ax.imshow(heatmap, alpha=0.5, cmap=cmap)
        plt.title(title)
        # Add the color scale bar on the right
        # plt.colorbar(im)
        # Add title

        plt.setp(ax, xticks=[], yticks=[])
        # show
        plt.show()

    def get_players_dists(self):
        """
        Calculate the distance each player moved
        """
        top_dist, top_dists_array = calculate_feet_dist(self.feet_top)
        bottom_dist, bottom_dists_array = calculate_feet_dist(self.feet_bottom)
        heatmap = self.get_player_position_heatmap(pit_size=10)
        heatmap[heatmap > 0] = 255
        heatmap = cv2.cvtColor(np.uint8(heatmap), cv2.COLOR_GRAY2BGR)
        heatmap[:, :, 1:] = 0

        self.display_heatmap(heatmap, self.court_tracker.court_reference.court, cmap=None, title='Players path')
        print('Top player distance is: {:.2f} m'.format(top_dist / 100))
        print('Bottom player distance is: {:.2f} m'.format(bottom_dist / 100))

        self.top_dists_array = top_dists_array
        self.bottom_dists_array = bottom_dists_array
        return top_dist, bottom_dist


def calculate_feet_dist(feet_positions, resolution=50):
    """
    Calculate feet positions for lower resolution
    """

    feet_positions = feet_positions // resolution
    feet_positions *= resolution
    total_dist = 0
    dists_array = [0]
    for pos1, pos2 in zip(feet_positions, feet_positions[1:]):
        dist = np.linalg.norm(pos1 - pos2)
        total_dist += dist
        dists_array.append(total_dist)
    return total_dist, dists_array


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
