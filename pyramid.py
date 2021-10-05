#!/bin/env python3
# import the necessary packages
import imutils

# This resizes the image
def pyramid(image, scale=1.2, minSize=(36, 36)):
	# yield the original image
	yield image
	# keep looping over the pyramid
	while True:
		# compute the new dimensions of the image and resize it
		w = int(image.shape[1] / scale)
		image = imutils.resize(image, width=w)
		# if the resized image does not meet the supplied minimum
		# size, then stop constructing the pyramid
		if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
			break
		# yield the next image in the pyramid
		yield image

def sliding_window(image, stepSize, windowSize):
	# slide a window across the image
	for y in range(0, image.shape[0], stepSize):
		for x in range(0, image.shape[1], stepSize):
			# yield the current window
			yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

# Load the image
# import time
# import cv2
# image = cv2.imread("./slowdive.jpg")
# winW, winH = (36, 36)


# loop over the image pyramid
# for resized in pyramid(image, scale=1.2):
	# loop over the sliding window for each layer of the pyramid
# 	for (x, y, window) in sliding_window(resized, stepSize=16, windowSize=(winW, winH)):
		# if the window does not meet our desired window size, ignore it
# 		if window.shape[0] != winH or window.shape[1] != winW:
# 			continue

# 		clone = resized.copy()
# 		cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
# 		cv2.imshow("Window", clone)
# 		cv2.waitKey(1)
# 		time.sleep(0.001)