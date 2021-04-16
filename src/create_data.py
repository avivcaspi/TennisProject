import imutils
import pandas as pd
from scipy import signal

import os
import cv2
import numpy as np
from src.detection import DetectionModel, center_of_box
from src.utils import get_dtype, get_video_properties


def convert_avi_to_mp4(avi_file_path, output_name):
    os.system("ffmpeg -i {input} -ac 2 -b:v 2000k -c:a aac -c:v libx264 -b:a 160k -vprofile high -bf 0 -strict experimental -f mp4 {output}".format(input = avi_file_path, output = output_name))
    os.remove(avi_file_path)

    return True


def create():
    dtype = get_dtype()

    df = pd.read_csv('../dataset/my_dataset/labels_data.csv')
    root, dirs, files = next(iter(os.walk("../dataset/my_dataset/patches")))
    files = [file.split('_')[0] + '_' + file.split('_')[1] + '.mp4' for file in files]
    df = df.loc[~df['name'].isin(files)]
    patch_size = 299
    df_misses = pd.read_csv('../dataset/my_dataset/misses.csv')
    for index, row in df_misses.iterrows():
        print(row['name'])
        filename = row['name']
        filepath = os.path.join('..', 'dataset','my_dataset', filename)

        detection_model = DetectionModel(dtype=dtype)

        video = cv2.VideoCapture(filepath)

        # get videos properties
        fps, length, v_width, v_height = get_video_properties(video)
        avi_file = os.path.join('..', 'dataset','my_dataset', 'patches', 'temp', os.path.splitext(filename)[0] + '.avi')
        mp4_file = os.path.join('..', 'dataset','my_dataset', 'patches', 'temp', os.path.splitext(filename)[0] + '_patch.mp4')
        out = cv2.VideoWriter(avi_file,
                              cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (patch_size, patch_size))

        frame_i = 0
        # Loop over all frames in the videos
        flag = True
        last_skipped = 0
        while True:
            ret, frame = video.read()
            frame_i += 1

            if ret:

                if frame_i == 1 or flag:
                    frame[:v_height//2, :, :] = [0,0,0]

                    '''cv2.imshow('res', frame)
                    if cv2.waitKey(0) & 0xff == 27:
                        cv2.destroyAllWindows()'''

                # detect
                res = detection_model.detect_player_1(frame.copy(), None)
                if res is None and flag:
                    print(f'Skipped frame {frame_i}')
                    last_skipped = frame_i
                    continue
                else:
                    flag = False

            else:
                break
        if detection_model.num_of_misses > 10:
            print(f'Many misses {detection_model.num_of_misses} file {filename}')
            '''line = {'name' : filename, 'misses': detection_model.num_of_misses}
            df_misses = df_misses.append(line, ignore_index=True)'''
            video.release()
            out.release()
            cv2.destroyAllWindows()
            continue
        if len(detection_model.player_1_boxes) > 0:
            positions = []
            for i, box in enumerate(detection_model.player_1_boxes):
                box_center = center_of_box(box)
                positions.append(box_center)
            positions = np.array(positions)
            smoothed = np.zeros_like(positions)
            smoothed[:, 0] = signal.savgol_filter(positions[:, 0], 7, 2)
            smoothed[:, 1] = signal.savgol_filter(positions[:, 1], 7, 2)
            box_margin = 150
            video = cv2.VideoCapture(filepath)
            video.set(1, last_skipped)
            for box_center in smoothed:
                ret, frame = video.read()

                if ret:

                    patch = frame[max(0,int(box_center[1] - box_margin)): min(frame.shape[0],int(box_center[1] + box_margin)),
                            max(0, int(box_center[0] - box_margin)): min(frame.shape[1],int(box_center[0] + box_margin))].copy()
                    if patch.shape[0] != patch.shape[1]:
                        min_size = min(patch.shape[:2])
                        patch = patch[:min_size, :min_size]
                    patch = imutils.resize(patch, patch_size)

                    out.write(patch)

                else:
                    break
            df_misses.loc[index, 'fixed'] = True
            print('Processing completed')
            video.release()
            out.release()
            cv2.destroyAllWindows()
            convert_avi_to_mp4(avi_file, mp4_file)
            if not index % 10 :
                df_misses.to_csv('../dataset/my_dataset/misses.csv', index=False)
            print('Saving done')

    df_misses.to_csv('../dataset/my_dataset/misses.csv', index=False)


csv_file = '../dataset/my_dataset/patches/labels.csv'
df = pd.read_csv(csv_file)
root, dirs, v_files = next(iter(os.walk("../dataset/my_dataset/patches")))
for i, row in df.iterrows():
    if row['name'] not in v_files:
        print(f'In df not in files : {row["name"]}')
k=0
for file in sorted(v_files):
    if file not in df['name'].values:
        print(f'In files not in df : {file}')
        k+= 1
print(k)