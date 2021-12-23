##
#   Author:		kostya
#   Created:	2021-12-23 11:36:15
#   Modified:	2021-12-23 12:21:16
##

import cv2
import numpy as np
import math
import sys
import shutil

def convolution(array, shape):
	Y, X = array.shape
	kernel_shape_x = np.linspace(0, X, shape[1] + 1)
	kernel_shape_y = np.linspace(0, Y, shape[0] + 1)

	conv_array = np.zeros(shape)

	for ker_x in range(len(kernel_shape_x) - 1):
		for ker_y in range(len(kernel_shape_y) - 1):
			kernel = array[
				math.floor(kernel_shape_y[ker_y]):math.floor(kernel_shape_y[ker_y + 1]),
				math.floor(kernel_shape_x[ker_x]):math.floor(kernel_shape_x[ker_x + 1])
			]

			# print(f'convolving {kernel.shape}')
			# print(kernel)
			conv_array[ker_y, ker_x] = kernel.sum() / (kernel.shape[0] * kernel.shape[1])
			# print(f'result = {conv_array[ker_y, ker_x]}')
			# print()
	return conv_array


def image2ascii(image):
	histogram = ' \'\";%$I@'
	
	image = image - image.min()
	ascii = np.floor(image * (len(histogram) / image.max())).astype(np.uint8)

	for i in range(len(histogram)):
		ascii[np.where(ascii == i)] = ord(histogram[i])
	return ascii


def frame_routine(image, shape):
	R = image[:, :, 0]
	G = image[:, :, 1]
	B = image[:, :, 2]

	# image = (R + G + B) / 3
	r_conv = convolution(R, shape)
	return r_conv


def main():
	fname = sys.argv[1]
	s = shutil.get_terminal_size()
	shape = [s.lines, s.columns]
	vidcap = cv2.VideoCapture(fname)
	FPS = 5

	i = 0
	while 1:
		success, image = vidcap.read()
		i += 1
		if FPS != i:
			continue
		i = 0
		if not success:
			break
		conv = frame_routine(image, shape)
		ascii = image2ascii(conv)
		for row in ascii:
			print(''.join(map(chr, row)))


def check():
	# a = np.arange(10).reshape((5, 2))
	# convolution(a, a.shape)
	a = np.arange(16).reshape((4, 4))
	convolution(a, (2, 2))


main()



# image = cv2.imread()
# while success:
  # cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file      
  # success, image = vidcap.read()
  # print('Read a new frame: ', success)
  # count += 1


def ps(s):
	for i in range(10):
		print(s * 10)
