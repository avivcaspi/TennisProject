import os
import time

import cv2
import numpy as np
from scipy import signal

from detection import DetectionModel
from pose import PoseExtractor
from smooth import Smooth
from src.shot_recognition import ActionRecognition
from utils import get_video_properties, get_dtype, get_stickman_line_connection
from court_detection import CourtDetector


def add_data_to_video(input_video, df, show_video, with_frame, output_folder,
                      output_file, stickman_pairs, court_detector):
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
    :param stickman_pairs: list, pairs of indices to create stickman figure
    :return: None
    """
    # landmarks colors
    circle_color, line_color = (0, 0, 255), (255, 0, 0)

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
        img_no_frame = np.zeros_like(img)

        # add pose stickman
        if df is not None:
            df = df.fillna(-1)
            values = np.array(df.values[frame_number], int)
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

        img_no_frame = court_detector.add_court_overlay(img_no_frame, frame_num=orig_frame)
        # display frame
        if show_video:
            cv2.imshow('Output-Skeleton', img)
        k = cv2.waitKey(1)
        if k == 27: break

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


def create_top_view(court_detector, player_1_boxes):
    court = court_detector.court_reference.court.copy()
    court = cv2.line(court, *court_detector.court_reference.net, 255, 5)
    inv_mat = court_detector.game_warp_matrix
    v_width, v_height = court.shape[::-1]
    court = cv2.cvtColor(court, cv2.COLOR_GRAY2BGR)
    out = cv2.VideoWriter('output/top_view.avi',
                          cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (v_width, v_height))
    positions = []
    for i, box in enumerate(player_1_boxes):
        feet_pos = np.array([(box[0] + (box[2] - box[0]) / 2).item(), box[3].item()]).reshape((1, 1, 2))
        feet_court_pos = cv2.perspectiveTransform(feet_pos, inv_mat[i]).reshape(-1)
        positions.append(feet_court_pos)
    positions = np.array(positions)
    smoothed = np.zeros_like(positions)
    smoothed[:, 0] = signal.savgol_filter(positions[:, 0], 7, 2)
    smoothed[:, 1] = signal.savgol_filter(positions[:, 1], 7, 2)

    for feet_pos in smoothed:
        frame = court.copy()
        frame = cv2.circle(frame, (int(feet_pos[0]), int(feet_pos[1])), 10, (0, 0, 255), 15)
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
    shot_recognition = ActionRecognition('saved_state_3e-5_86%')

    # Load videos from videos path
    video = cv2.VideoCapture(video_path)

    # get videos properties
    fps, length, v_width, v_height = get_video_properties(video)

    # Output videos writer
    out = cv2.VideoWriter(os.path.join(output_folder, output_file + '.avi'),
                          cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (v_width, v_height))

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

            '''if court_detector.check_court_movement(frame):
                court_detector.detect(frame)'''

            # initialize landmarks lists
            stickman_marks = np.zeros_like(frame)

            # detect
            boxes = detection_model.detect_player_1(frame.copy(), court_detector)

            # Create stick man figure (pose detection)
            if stickman:
                stickman_marks = pose_extractor.extract_pose(frame, detection_model.player_1_boxes)

            #probs, stroke = shot_recognition.predict_stroke(frame, detection_model.player_1_boxes[-1])
            # Combine all landmarks
            # TODO clean this shit
            total_marks = np.clip(stickman_marks + boxes, 0, 255)
            mask = np.sum(total_marks, axis=2)
            frame[mask != 0, :] = [0,0,0]
            frame = frame + total_marks if include_video else total_marks
            '''cv2.putText(frame, 'Forehand - {:.2f}, Backhand - {:.2f}, Service - {:.2f}'.format(*probs),
                        (100, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
            cv2.putText(frame, f'Stroke : {stroke}',
                        (100, 160),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)'''

            if court:
                frame = court_detector.add_court_overlay(frame, overlay_color=(0, 0, 255))
            # Output frame and save it
            if show_video:
                cv2.imshow('frame', frame)
            # cv2.imwrite('../report/persons_detections_4.png', frame)
            out.write(frame)
            total_time += (time.time() - start_time)
            print('Processing frame %d/%d  FPS %04f' % (frame_i, length, frame_i / total_time), '\r', end='')
            if not frame_i % 100:
                print('')
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    print('Processing frame %d/%d  FPS %04f' % (length, length, length / total_time), '\n', end='')
    print('Processing completed')
    video.release()
    out.release()
    cv2.destroyAllWindows()
    if top_view:
        create_top_view(court_detector, detection_model.player_1_boxes)
    # Save landmarks in csv files
    df = None
    # Save stickman data
    if stickman:
        df = pose_extractor.save_to_csv(output_folder)

    # smooth the output data for better results
    if smoothing:
        smoother = Smooth()
        df_smooth = smoother.smooth(df)
        smoother.save_to_csv(output_folder)

        smoothing_output_file = output_file + '_smoothing'
        # add smoothing data to the videos
        add_data_to_video(video_path, df_smooth, show_video, 2, output_folder,
                          smoothing_output_file, get_stickman_line_connection(), court_detector)


s = time.time()
video_process(video_path='../videos/vid1.mp4', show_video=True, stickman=True, stickman_box=False, smoothing=True,
              court=False, top_view=False)
print(f'Total computation time : {time.time() - s} seconds')
