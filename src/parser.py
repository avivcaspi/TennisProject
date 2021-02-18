import os
import argparse
from pathlib import Path
from process import video_process
from utils import str2bool

if __name__ == '__main__':
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--videos", required=True,
                    help="str, path to input videos")
    ap.add_argument("--show-videos", required=False, default=False, type=str2bool,
                    help="bool, show videos while processing (default=False)")
    ap.add_argument("-s", "--stickman", required=False, default=True, type=str2bool,
                    help="bool, calculate figure's stickman (default=True)")
    ap.add_argument("--stickman-box", required=False, default=False, type=str2bool,
                    help="bool, show stickman box (default=False)")
    ap.add_argument("-o", "--output-file", required=False, default='output_video',
                    help="str, output file name for videos output (default='output_video')")
    ap.add_argument("--output-folder", required=False, default='output_data',
                    help="str, path to results csv files folder (default='output_data')")
    ap.add_argument("-w", "--width", required=False, default=500, type=int,
                    help="int, output videos width (height will be determine by crop-dim value and width to preserve ratio)\
                     (default=500)")
    ap.add_argument("--crop-center", required=False, default=True, type=str2bool,
                    help="bool, determine if we crop the center of the videos to get 1:1 width height ratio ("
                         "default=True)")
    ap.add_argument("--smoothing", required=False, default=True, type=str2bool,
                    help="bool, determine if we use smoothing on the output (default=True)")
    args = vars(ap.parse_args())

    Path(args['output_folder']).mkdir(parents=True, exist_ok=True)
    video_process(video_path=args['videos'], show_video=args['show_video'], include_video=True,
                  stickman=args['stickman'], stickman_box=args['stickman_box'],
                  output_file=args['output_file'], output_folder=args['output_folder'],
                  width=args['width'], crop_dim=args['crop_center'], smoothing=args['smoothing'])