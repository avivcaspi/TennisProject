import numpy as np
import cv2
import itertools
import csv
from collections import defaultdict
import torch
import random

# get input array
def getInputArr(path, path1, path2, width, height):
	try:
		# read the image
		img = cv2.imread(path, 1)
		# resize it
		img = cv2.resize(img, (width, height))
		# input must be float type
		img = img.astype(np.float32)

		# read the image
		img1 = cv2.imread(path1, 1)
		# resize it
		img1 = cv2.resize(img1, (width, height))
		# input must be float type
		img1 = img1.astype(np.float32)

		# read the image
		img2 = cv2.imread(path2, 1)
		# resize it
		img2 = cv2.resize(img2, (width, height))
		# input must be float type
		img2 = img2.astype(np.float32)

		# combine three imgs to  (width , height, rgb*3)
		imgs = np.concatenate((img, img1, img2), axis=2)

		# since the odering of TrackNet  is 'channels_first', so we need to change the axis
		imgs = np.rollaxis(imgs, 2, 0)
		return imgs

	except Exception as e:

		print(path, e)


# get output array
def getOutputArr(path, nClasses, width, height):
	seg_labels = np.zeros((height, width, nClasses))
	try:
		img = cv2.imread(path, 1)
		img = cv2.resize(img, (width, height))
		img = img[:, :, 0]

		for c in range(nClasses):
			seg_labels[:, :, c] = (img == c).astype(int)

	except Exception as e:
		print(e)

	seg_labels = np.reshape(seg_labels, (width * height, nClasses))
	seg_labels = seg_labels.transpose([1,0]).argmax(0)
	return seg_labels


# read input data and output data
def InputOutputGenerator(images_path, batch_size, n_classes, input_height, input_width, output_height, output_width):
	# read csv file to 'zipped'
	columns = defaultdict(list)
	rows = []
	with open(images_path) as f:
		reader = csv.reader(f)
		next(reader)
		for row in reader:
			rows.append(row)
	random.shuffle(rows)
	zipped = itertools.cycle(rows)

	while True:
		Input = []
		Output = []
		# read input&output for each batch
		for _ in range(batch_size):
			path, path1, path2, anno = next(zipped)
			Input.append(getInputArr(path, path1, path2, input_width, input_height))
			Output.append(getOutputArr(anno, n_classes, output_width, output_height))
		# return input&output
		yield torch.from_numpy(np.array(Input)) / 255, torch.from_numpy(np.array(Output))


'''for x, y in InputOutputGenerator('../dataset/Dataset/training_model2.csv', 1, 256, 360, 640, 360, 640):
	y = y.argmax(dim=2).reshape([1, 360,640])
	cv2.imshow('g', np.uint8(y[0]))
	if cv2.waitKey(0) & 0xff == 27:
		cv2.destroyAllWindows()
	print('sdf')
'''