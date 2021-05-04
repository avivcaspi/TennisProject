import os
import cv2
import torch
import torchvision
import numpy as np
import pandas as pd


class PoseExtractor:
    def __init__(self, person_num=1, box=False, dtype=torch.FloatTensor):
        """
        Extractor for pose keypoints
        :param person_num: int, number of person in the videos (default = 1)
        :param box: bool, show person bounding box in the output frame (default = False)
        :param dtype: torch.type, dtype of the mdoel and image, determine if we use GPU or not
        """
        self.pose_model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True)
        self.pose_model.type(dtype)  # Also moves model to GPU if available
        self.pose_model.eval()
        self.dtype = dtype
        self.person_num = person_num
        self.box = box
        self.PERSON_LABEL = 1
        self.SCORE_MIN = 0.9
        self.keypoint_threshold = 2
        self.data = []
        self.line_connection = [(7, 9), (7, 5), (10, 8), (8, 6), (6, 5), (15, 13),
                                (13, 11), (11, 12), (12, 14), (14, 16), (5, 11), (12, 6)]
        self.COCO_PERSON_KEYPOINT_NAMES = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder', 'left_elbow',
            'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee', 'right_knee',
            'left_ankle', 'right_ankle'
        ]

    def _add_lines(self, frame, keypoints, keypoints_scores):
        # Add line using the keypoints connections to create stick man
        for a, b in self.line_connection:
            if keypoints_scores[a] > self.keypoint_threshold and keypoints_scores[b] > self.keypoint_threshold:
                p1 = (int(keypoints[a][0]), int(keypoints[a][1]))
                p2 = (int(keypoints[b][0]), int(keypoints[b][1]))
                cv2.line(frame, p1, p2, [0, 0, 255], 2)
        # Connect nose to center of torso
        a = 0
        p1 = (int(keypoints[a][0]), int(keypoints[a][1]))
        p2 = (int((keypoints[5][0] + keypoints[6][0]) / 2), int((keypoints[5][1] + keypoints[6][1]) / 2))
        cv2.line(frame, p1, p2, [0, 0, 255], 2)
        return frame

    def extract_pose(self, image, player_boxes):
        """
        extract pose from given image using pose_model
        :param player_boxes:
        :param image: ndarray, the image we would like to extract the pose from
        :return: frame that include the pose stickman
        """
        height, width = image.shape[:2]
        if len(player_boxes) > 0:
            margin = 50
            xt, yt, xb, yb = player_boxes[-1]
            xt, yt, xb, yb = int(xt), int(yt), int(xb), int(yb)
            patch = image[max(yt - margin, 0):min(yb + margin, height), max(xt - margin, 0):min(xb + margin, width)].copy()
        else:
            margin = 0
            xt, yt, xb, yb = 0, 0, width, height
            patch = image.copy()
        # initialize pose stickman frame and data
        stickman = np.zeros_like(image)
        patch_zeros = np.zeros_like(patch)
        x_data, y_data = [], []

        # creating torch.tensor from the image ndarray
        frame_t = patch.transpose((2, 0, 1)) / 255
        frame_tensor = torch.from_numpy(frame_t).unsqueeze(0).type(self.dtype)

        # Finding boxes and keypoints
        with torch.no_grad():
            # forward pass
            p = self.pose_model(frame_tensor)

        # add bounding box for each person found
        if self.box:
            # Marking every person found in the image with high score
            for box, label, score in zip(p[0]['boxes'][:self.person_num], p[0]['labels'], p[0]['scores']):
                if label == self.PERSON_LABEL and score > self.SCORE_MIN:
                    cv2.rectangle(patch_zeros, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), [0, 0, 255], 2)

        # Marking all keypoints of the person we found, and connecting part to create the stick man
        for keypoints, keypoint_scores, score in zip(p[0]['keypoints'][:self.person_num], p[0]['keypoints_scores'],
                                                     p[0]['scores']):
            # only find person with high score
            if score > self.SCORE_MIN:
                for i, ((x, y, v), key_point_score) in enumerate(zip(keypoints, keypoint_scores)):
                    # add keypoint only if it exceed threshold score
                    if key_point_score > self.keypoint_threshold:
                        x_data.append(x.item() + max(xt - margin, 0))
                        y_data.append(y.item() + max(yt - margin, 0))
                        cv2.circle(patch_zeros, (int(x), int(y)), 2, [255, 0, 0], 2)
                    else:
                        # if the keypoint was not found we add None
                        # in the smoothing section we will try to complete the missing data
                        x_data.append(None)
                        y_data.append(None)
                # create the stickman using the keypoints we found
                self._add_lines(patch_zeros, keypoints, keypoint_scores)
            self.data.append(x_data + y_data)
        stickman[max(yt - margin, 0):min(yb + margin, height), max(xt - margin, 0):min(xb + margin, width)] = patch_zeros

        return stickman

    def save_to_csv(self, output_folder):
        """
        Saves the pose keypoints data as csv
        :param output_folder: str, path to output folder
        :return: df, the data frame of the pose keypoints
        """
        columns = self.COCO_PERSON_KEYPOINT_NAMES
        columns_x = [column + '_x' for column in columns]
        columns_y = [column + '_y' for column in columns]
        df = pd.DataFrame(self.data, columns=columns_x + columns_y)
        outfile_path = os.path.join(output_folder, 'stickman_data.csv')
        df.to_csv(outfile_path, index=False)
        return df
