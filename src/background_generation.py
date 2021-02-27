import cv2
import numpy as np


class BackgroundGenerator:
    def __init__(self):
        self.background = None
        self.bg_flags = None
        self.blocks_left = 0
        self.block_width = 64
        self.block_height = 8
        self.block_area = self.block_width * self.block_height
        self.width = 0
        self.height = 0
        self.max_difference = 3
        self.min_percentage = 0.98
        self.last_frame = None
        self.bg_generate_complete = False

    def generate_bg(self, frame):
        frame = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2GRAY)
        if self.background is None:
            self.height, self.width = frame.shape[:2]
            self.background = np.zeros((self.height, self.width), dtype=np.uint8)
            col_blocks = int(np.ceil(self.height / self.block_height))
            row_blocks = int(np.ceil(self.width / self.block_width))
            self.bg_flags = np.zeros((col_blocks, row_blocks))
            self.blocks_left = (self.height // self.block_height) * (self.width // self.block_width)
        else:
            for block_x in range(0, self.width, self.block_width):
                for block_y in range(0, self.height, self.block_height):
                    flag_x, flag_y = int(np.ceil(block_x / self.block_width)),\
                                     int(np.ceil(block_y / self.block_height))
                    if not self.bg_flags[flag_y, flag_x]:
                        block1 = self.last_frame[block_y:min(block_y + self.block_height, self.height),
                                 block_x: min(block_x + self.block_width, self.width)]
                        block2 = frame[block_y:min(block_y + self.block_height, self.height),
                                 block_x: min(block_x + self.block_width, self.width)]
                        is_stable = self._is_stable_block(block1, block2)
                        if is_stable:
                            self.background[block_y:min(block_y + self.block_height, self.height),
                            block_x: min(block_x + self.block_width, self.width)] = block1.copy()
                            self.bg_flags[flag_y, flag_x] = True
                            self.blocks_left -= 1
                            if self.blocks_left == 0:
                                self.bg_generate_complete = True
                                print(f'finished bg gen')
                                cv2.imshow('bg', self.background)

                                if cv2.waitKey(0) & 0xff == 27:
                                    cv2.destroyAllWindows()
                                return True
        self.last_frame = frame
        return False

    def _is_stable_block(self, block1, block2):
        diff = abs(block1 - block2)
        num_stable = np.sum(diff < self.max_difference)
        is_stable = num_stable > self.min_percentage * self.block_area
        return is_stable


if __name__ == '__main__':
    bg_gen = BackgroundGenerator()
    video = cv2.VideoCapture('../videos/vid5.mp4')
    # Loop over all frames in the videos
    frame_num = 0
    while True:
        ret, frame = video.read()
        frame_num += 1
        if ret:
            finished = bg_gen.generate_bg(frame)

            if finished:
                bg = bg_gen.background.copy()
                print(f'finished in {frame_num} frames')
                cv2.imshow('bg', bg)

                if cv2.waitKey(0) & 0xff == 27:
                    cv2.destroyAllWindows()
                break
        else:
            break
    video.release()
    cv2.destroyAllWindows()
