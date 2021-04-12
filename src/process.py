import os
import time

import cv2
import numpy as np
from scipy import signal
from scipy.interpolate import interp1d
from scipy.signal import find_peaks

from detection import DetectionModel, center_of_box
from pose import PoseExtractor
from smooth import Smooth
from src.ball_detection import BallDetector
from src.shot_recognition import ActionRecognition
from utils import get_video_properties, get_dtype, get_stickman_line_connection
from court_detection import CourtDetector
import matplotlib.pyplot as plt


def plot_player_ball_dist(player_boxes, ball_positions, skeleton_df, title):

    ball_x, ball_y = ball_positions[:, 0], ball_positions[:, 1]
    smooth_x = signal.savgol_filter(ball_x, 3, 2)
    smooth_y = signal.savgol_filter(ball_y, 3, 2)

    x = np.arange(0, len(smooth_y))
    indices = [i for i, val in enumerate(smooth_y) if np.isnan(val)]
    x = np.delete(x, indices)
    y1 = np.delete(smooth_y, indices)
    y2 = np.delete(smooth_x, indices)
    f2_y = interp1d(x, y1, kind='cubic', fill_value="extrapolate")
    f2_x = interp1d(x, y2, kind='cubic', fill_value="extrapolate")
    xnew = np.linspace(0, len(ball_y), num=len(ball_y), endpoint=True)
    plt.plot(np.arange(0, len(smooth_y)), smooth_y, 'o', xnew,
             f2_y(xnew), '-r')
    plt.legend(['data', 'inter'], loc='best')
    plt.show()

    positions = f2_y(xnew)
    peaks, _ = find_peaks(positions)
    plt.plot(positions)
    plt.plot(peaks, positions[peaks], "x")
    plt.show()

    left_wrist_index = 9
    right_wrist_index = 10
    skeleton_df = skeleton_df.fillna(-1)
    left_wrist_pos = skeleton_df.iloc[:, [left_wrist_index, left_wrist_index + 15]].values
    right_wrist_pos = skeleton_df.iloc[:, [right_wrist_index, right_wrist_index + 15]].values

    dists = []
    for i, player_box in enumerate(player_boxes):
        if player_box[0] is not None:
            player_center = center_of_box(player_box)
            ball_pos = np.array([f2_x(i), f2_y(i)])
            box_dist = np.linalg.norm(player_center - ball_pos)
            right_wrist_dist, left_wrist_dist = np.inf, np.inf
            if right_wrist_pos[i,0] > 0:
                right_wrist_dist = np.linalg.norm(right_wrist_pos[i, :] - ball_pos)
            if left_wrist_pos[i, 0] > 0:
                left_wrist_dist = np.linalg.norm(left_wrist_pos[i, :] - ball_pos)
            dists.append(min(box_dist, right_wrist_dist, left_wrist_dist))
        else:
            dists.append(None)
    dists = np.array(dists)

    strokes_indices = []
    for peak in peaks:
        print(peak, dists[peak])
        if dists[peak] < 100:
            strokes_indices.append(peak)
    return strokes_indices


def mark_player_box(frame, boxes, frame_num):
    box = boxes[frame_num]
    if box[0] is not None:
        cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), [255, 0, 255], 2)
    return frame


def mark_skeleton(skeleton_df, img, img_no_frame, frame_number):
    # landmarks colors
    circle_color, line_color = (0, 0, 255), (255, 0, 0)
    stickman_pairs = get_stickman_line_connection()

    skeleton_df = skeleton_df.fillna(-1)
    values = np.array(skeleton_df.values[frame_number], int)
    points = list(zip(values[5:17], values[22:]))
    # draw key points
    for point in points:
        if point[0] >= 0 and point[1] >= 0:
            xy = tuple(np.array([point[0], point[1]], int))
            cv2.circle(img, xy, 2, circle_color, 2)
            cv2.circle(img_no_frame, xy, 2, circle_color, 2)

    # Draw stickman
    for pair in stickman_pairs:
        partA = pair[0] - 5
        partB = pair[1] - 5
        if points[partA][0] >= 0 and points[partA][1] >= 0 and points[partB][0] >= 0 and points[partB][1] >= 0:
            cv2.line(img, points[partA], points[partB], line_color, 1, lineType=cv2.LINE_AA)
            cv2.line(img_no_frame, points[partA], points[partB], line_color, 1, lineType=cv2.LINE_AA)
    return img, img_no_frame


def add_data_to_video(input_video, court_detector, players_detector, ball_detector, shot_recognition, skeleton_df,
                      show_video, with_frame, output_folder, output_file):
    """
    Creates new videos with pose stickman, face landmarks and blinks counter
    :param input_video: str, path to the input videos
    :param df: DataFrame, data of the pose stickman positions
    :param show_video: bool, display output videos while processing
    :param with_frame: int, output videos includes the original frame with the landmarks
    (0 - only landmarks, 1 - original frame with landmarks, 2 - original frame with landmarks and only
    landmarks (side by side))
    :param output_folder: str, path to output folder
    :param output_file: str, name of the output file
    :return: None
    """

    player1_boxes = players_detector.player_1_boxes
    player2_boxes = players_detector.player_2_boxes

    strokes_indices = plot_player_ball_dist(player1_boxes, ball_detector.xy_coordinates, skeleton_df,'Bottom Player')
    # plot_player_ball_dist(player2_boxes, ball_detector.xy_coordinates, 'Top Player')

    if skeleton_df is not None:
        skeleton_df = skeleton_df.fillna(-1)

    # Read videos file
    cap = cv2.VideoCapture(input_video)

    # videos properties
    fps, length, width, height = get_video_properties(cap)

    final_width = width * 2 if with_frame == 2 else width

    # Video writer
    out = cv2.VideoWriter(os.path.join(output_folder, output_file + '.avi'),
                          cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (final_width, height))

    # initialize frame counters
    frame_number = 0
    orig_frame = 0
    while True:
        orig_frame += 1
        print('Creating new videos frame %d/%d  ' % (orig_frame, length), '\r', end='')
        if not orig_frame % 100:
            print('')
        ret, img = cap.read()

        if not ret:
            break

        # initialize frame for landmarks only
        img_no_frame = np.ones_like(img) * 255

        # add Court location
        img = court_detector.add_court_overlay(img, overlay_color=(0, 0, 255), frame_num=frame_number)
        img_no_frame = court_detector.add_court_overlay(img_no_frame, overlay_color=(0, 0, 255), frame_num=frame_number)

        # add players locations
        img = mark_player_box(img, player1_boxes, frame_number)
        img = mark_player_box(img, player2_boxes, frame_number)
        img_no_frame = mark_player_box(img_no_frame, player1_boxes, frame_number)
        img_no_frame = mark_player_box(img_no_frame, player2_boxes, frame_number)

        # add ball location
        img = ball_detector.mark_positions(img, frame_num=frame_number)
        img_no_frame = ball_detector.mark_positions(img_no_frame, frame_num=frame_number, ball_color='black')

        # add pose stickman
        if skeleton_df is not None:
            img, img_no_frame = mark_skeleton(skeleton_df, img, img_no_frame, frame_number)

        for i in range(-5,5):
            if frame_number + i in strokes_indices:
                cv2.putText(img, 'STROKE HIT', (200, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 3)
                break
        '''cv2.putText(img, f'Dist {dists[frame_number]}', (200, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 3)'''
        # display frame
        if show_video:
            cv2.imshow('Output', img)
            if cv2.waitKey(1) & 0xff == 27:
                cv2.destroyAllWindows()

        # save output videos
        if with_frame == 0:
            final_frame = img_no_frame
        elif with_frame == 1:
            final_frame = img
        else:
            final_frame = np.concatenate([img, img_no_frame], 1)
        out.write(final_frame)
        frame_number += 1
    print('Creating new videos frame %d/%d  ' % (length, length), '\n', end='')
    print(f'New videos created, file name - {output_file}.avi')
    cap.release()
    out.release()
    cv2.destroyAllWindows()


def create_top_view(court_detector, player_1_boxes, player_2_boxes):
    court = court_detector.court_reference.court.copy()
    court = cv2.line(court, *court_detector.court_reference.net, 255, 5)
    inv_mat = court_detector.game_warp_matrix
    v_width, v_height = court.shape[::-1]
    court = cv2.cvtColor(court, cv2.COLOR_GRAY2BGR)
    out = cv2.VideoWriter('output/top_view.avi',
                          cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (v_width, v_height))
    positions_1 = []
    positions_2 = []
    for i, box in enumerate(player_1_boxes):
        feet_pos = np.array([(box[0] + (box[2] - box[0]) / 2).item(), box[3].item()]).reshape((1, 1, 2))
        feet_court_pos = cv2.perspectiveTransform(feet_pos, inv_mat[i]).reshape(-1)
        positions_1.append(feet_court_pos)
    mask = []
    for i, box in enumerate(player_2_boxes):
        if box[0] is not None:
            feet_pos = np.array([(box[0] + (box[2] - box[0]) / 2), box[3]]).reshape((1, 1, 2))
            feet_court_pos = cv2.perspectiveTransform(feet_pos, inv_mat[i]).reshape(-1)
            positions_2.append(feet_court_pos)
            mask.append(True)
        elif len(positions_2) > 0:
            positions_2.append(positions_2[-1])
            mask.append(False)
        else:
            positions_2.append(np.array([0, 0]))
            mask.append(False)

    positions_1 = np.array(positions_1)
    smoothed_1 = np.zeros_like(positions_1)
    smoothed_1[:, 0] = signal.savgol_filter(positions_1[:, 0], 7, 2)
    smoothed_1[:, 1] = signal.savgol_filter(positions_1[:, 1], 7, 2)
    positions_2 = np.array(positions_2)
    smoothed_2 = np.zeros_like(positions_2)
    smoothed_2[:, 0] = signal.savgol_filter(positions_2[:, 0], 7, 2)
    smoothed_2[:, 1] = signal.savgol_filter(positions_2[:, 1], 7, 2)

    smoothed_2[not mask, :] = [None, None]

    for feet_pos_1, feet_pos_2 in zip(smoothed_1, smoothed_2):
        frame = court.copy()
        frame = cv2.circle(frame, (int(feet_pos_1[0]), int(feet_pos_1[1])), 10, (0, 0, 255), 15)
        if feet_pos_2[0] is not None:
            frame = cv2.circle(frame, (int(feet_pos_2[0]), int(feet_pos_2[1])), 10, (0, 0, 255), 15)
        out.write(frame)
    out.release()
    cv2.destroyAllWindows()


def video_process(video_path, show_video=False, include_video=True,
                  stickman=True, stickman_box=True, court=True,
                  output_file='output', output_folder='output',
                  smoothing=True, top_view=True):
    """
    Takes videos of one person as input, and calculate the body pose and face landmarks, and saves them as csv files.
    Also, output a result videos with the keypoints marked.
    :param court:
    :param video_path: str, path to the videos
    :param show_video: bool, show processed videos while processing (default = False)
    :param include_video: bool, result output videos will include the original videos as well as the
    keypoints (default = True)
    :param stickman: bool, calculate pose and create stickman using the pose data (default = True)
    :param stickman_box: bool, show person bounding box in the output videos (default = False)
    :param output_file: str, output file name (default = 'output')
    :param output_folder: str, output folder name (default = 'output') will create new folder if it does not exist
    :param smoothing: bool, use smoothing on output data (default = True)
    :return: None
    """
    dtype = get_dtype()

    # initialize extractors
    court_detector = CourtDetector()
    detection_model = DetectionModel(dtype=dtype)
    pose_extractor = PoseExtractor(person_num=1, box=stickman_box, dtype=dtype) if stickman else None
    shot_recognition = ActionRecognition('saved_state_strokes_3e-05_50%_labels')
    ball_detector = BallDetector('saved states/tracknet_weights_lr_1.0_epochs_150_last_trained.pth', out_channels=2)

    # Load videos from videos path
    video = cv2.VideoCapture(video_path)

    # get videos properties
    fps, length, v_width, v_height = get_video_properties(video)

    # frame counter
    frame_i = 0

    # time counter
    total_time = 0

    # Loop over all frames in the videos
    while True:
        start_time = time.time()
        ret, frame = video.read()
        frame_i += 1

        if ret:
            if frame_i == 1:
                court_detector.detect(frame)
                print(f'Court detection {"Success" if court_detector.success_flag else "Failed"}')
                print(f'Time to detect court :  {time.time() - start_time} seconds')
                start_time = time.time()

            court_detector.track_court(frame)

            # detect
            detection_model.detect_player_1(frame.copy(), court_detector)
            detection_model.detect_top_persons(frame, court_detector, frame_i)

            # Create stick man figure (pose detection)
            if stickman:
                pose_extractor.extract_pose(frame, detection_model.player_1_boxes)

            ball_detector.detect_ball(court_detector.delete_extra_parts(frame))

            # probs, stroke = shot_recognition.predict_stroke(frame, detection_model.player_1_boxes[-1])

            total_time += (time.time() - start_time)
            print('Processing frame %d/%d  FPS %04f' % (frame_i, length, frame_i / total_time), '\r', end='')
            if not frame_i % 100:
                print('')
        else:
            break
    print('Processing frame %d/%d  FPS %04f' % (length, length, length / total_time), '\n', end='')
    print('Processing completed')
    video.release()
    cv2.destroyAllWindows()

    detection_model.find_player_2_box()

    if top_view:
        create_top_view(court_detector, detection_model.player_1_boxes, detection_model.player_2_boxes)

    # Save landmarks in csv files
    df = None
    # Save stickman data
    if stickman:
        df = pose_extractor.save_to_csv(output_folder)

    # smooth the output data for better results
    df_smooth = None
    if smoothing:
        smoother = Smooth()
        df_smooth = smoother.smooth(df)
        smoother.save_to_csv(output_folder)

    add_data_to_video(input_video=video_path, court_detector=court_detector, players_detector=detection_model,
                      ball_detector=ball_detector, shot_recognition=shot_recognition, skeleton_df=df_smooth,
                      show_video=show_video, with_frame=1, output_folder=output_folder, output_file=output_file)

    ball_detector.show_y_graph(detection_model.player_1_boxes, detection_model.player_2_boxes)


s = time.time()
video_process(video_path='../videos/vid1.mp4', show_video=True, stickman=True, stickman_box=False, smoothing=True,
              court=True, top_view=False)
print(f'Total computation time : {time.time() - s} seconds')
