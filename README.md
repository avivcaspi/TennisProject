# Tennis Project

## Intro
This project is the final project for B.Sc degree at the Technion institute.
In this project I present an end-to-end model that uses computer vision algorithms and deep learning networks in order to process and analyse
official tennis matches.
Main paper used for this project is [Vinyes-Mora-S-2018-PhD-Thesis](https://spiral.imperial.ac.uk/handle/10044/1/67949)

## Project Scope
This project is mainly focused on official match videos, and camera position and angle as shown in the next section.

Example for some tasks completed in this project are:
1. Detecting and tracking the tennis court in frame
2. Detecting and tracking both player in the frame 
3. Detecting and tracking the ball 
4. Extract the bottom player skeleton
5. Detecting the exact moment of strokes 
6. Classify the strokes of the bottom player
7. Create top view gameplay
8. Calculate statistics of the gameplay of both players including positions heatmap and distance travelled

## Example
![alt text](https://github.com/avivcaspi/TennisProject/blob/main/example_short.gif)

## Requirements
Python 3.4, OpenCV (cv2), PyTorch and other common packages listed in `requirements.txt` at the [src](https://github.com/avivcaspi/TennisProject/blob/main/src) directory.

## Entry point
Entry point for the code is in process.py (scroll to the bottom to see main function video_process(...)).
