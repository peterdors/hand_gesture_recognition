# Peter Dorsaneo

# main.py
# ========
# Python script for simple hand gesture recognition. This doesn't necessarily 
# recognize if a hand is in the image or not. It is just for determining on  
# images of hands if they are holding either 0, 1, 2, 3, 4, or 5 fingers up. 

# USAGE: python main.py -i path/to/image/file

# TODO(s): 	Get gesture recognition accurate for video processing of hand 
# 			gestures. 

import cv2
import argparse
import os
import numpy as np
import math
from time import sleep

# @ return values: boolean, thresh_img
def get_threshold(image, thresh=0, maxval=255, t=cv2.THRESH_BINARY):
	return cv2.threshold(image, thresh, maxval, t)

# @ return values: contours_in_image, hierarchy_of_contours
def find_contours(threshold):
	# Use of CHAIN_APPROX_NONE is a memory hog but it is very useful
	# for our purposes.
	return cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

# @ return values: image
def draw_contour(image, contour, color=(0,255,0), line_thickness=5):
	return cv2.drawContours(image, contour, -1, color, line_thickness)

# @ return values: image
# To put our process for drawing contours into a simple function. 
def get_contours(mask, image):
	# We need to use the gray scaled image for determining the threshold. 
	ret, thresh = get_threshold(mask)

	# Using the threshold created above, we will find the contours of the image. 
	contours, hierarchy = find_contours(thresh)

	return draw_contour(image, contours)

# @ return values: float
# simple distance calculation between two points on an xy-plane. 
def calculate_distance(x1, x2, y1, y2):
	return math.sqrt(abs(x1 - x2) ** 2 + abs(y1 - y2) ** 2)

# @ return values: float
# Gets the angle between line segments denoted as <ab> and <bc>
def calculate_angle(a, b, c):
	# Get the distance from each side of the triangle shape.
	ab = math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
	bc = math.sqrt((b[0] - c[0])**2 + (b[1] - c[1])**2)
	ac = math.sqrt((a[0] - c[0])**2 + (a[1] - c[1])**2)

	# Float value when we take (180 / pi). 
	rad_to_deg = 57.2957795

	# Converts from radians output to a degrees output. 
	return math.acos((bc**2 + ac**2 - ab**2) / (2 * bc * ac)) * rad_to_deg



# Parameters: frame - the captured image to be processed.
# @ return values: The original image with text describing the gesture. 
def gesture_recognizer(frame): 
	# We don't necessarily need to read the frame again since it is already 
	# to be assumed it was read in before the function call. 
	# image = cv2.imread(frame, cv2.IMREAD_COLOR)
	frame_size = 640

	image = cv2.resize(frame, (frame_size, frame_size))
	orig_image = image.copy()

	# For filtering out any noise in the image.
	image_blur = cv2.blur(image, (10,10))

	# Convert the image to HSV colors for skin segmentation. 
	image_hsv = cv2.cvtColor(image_blur, cv2.COLOR_BGR2HSV)

	# HSV values: Hue, Saturation, Vibrance(?)
	# Our bounds for my skin color (Caucasian).
	lower_skin_color = np.array([0, 0.28 * 255, 0])
	upper_skin_color = np.array([50, 0.68 * 255, 255])

	# Find the skin color in the HSV colored image. 
	mask = cv2.inRange(image_hsv, lower_skin_color, upper_skin_color)

	# We need to use the gray scaled image for determining the threshold. 
	ret, thresh = get_threshold(mask)

	# Using the threshold created above, we will find the contours of the image. 
	contours, hierarchy = find_contours(thresh)

	# Get the largest contour in the frame, the largest contour *should* be the 
	# hand. 
	contour = max(contours, key=cv2.contourArea)

	image = draw_contour(image, contour)

	# This gives us the indice for the (x,y) coordinate point in the
	# contour variable 
	hulls = cv2.convexHull(contour, returnPoints=False)

	# Here we will compute the convexity defects of the hand.
	# Also draw them on the image.  
	defects = cv2.convexityDefects(contour, hulls)

	max_dist = 15
	radius = 10
	thickness = 2

	# I hate running this as a while loop and doing the first iteration manually
	# but it was a quick and easy way to get things going while also using my 
	# method of saving the (prev)ious convexity defect data.

	s,e,f,d = defects[0,0]
	prev = tuple(contour[f][0])
	cv2.circle(image, prev, radius, (255,255,255), thickness)
	
	i = 1
	cnt = 0
	cnt_used_defects = 0
	while i < defects.shape[0]:
		s,e,f,d = defects[i,0]
		i += 1
		start = tuple(contour[s][0])
		end = tuple(contour[e][0])
		far = tuple(contour[f][0])

		# Prevents evaluating for points that are within a certain distance
		# as these would be determined to be points on the same finger or 
		# crevice. 
		if (calculate_distance(prev[0], far[0], prev[1], far[1]) < max_dist):
			continue

		# Another cheeky way of making sure not to count the points that occur 
		# at the bottom of the image. These points are irrelevant. 
		if (start[1] > 600 or end[1] > 600 or far[1] > 600): 
			continue

		cnt_used_defects += 1
		# Just for drawing lines between the finger tip and the crevice between 
		# two fingers (ideally).
		cv2.line(image, start, end, (255, 255, 0), 2)
		cv2.line(image, start, far, (255, 255, 0), 2)
		cv2.line(image, far, end, (255, 255, 0), 2)

		cv2.circle(image, far, radius, (255,255,255), thickness)

		# Find the angles for every point referenced. 
		# a: start, b: end, c: far
		gamma = calculate_angle(start, end, far)

		# Two fingers will usually make an angle within 90 degrees so this can 
		# be our counting point.  
		if (gamma <= 90):
			# print(start, end, far, gamma)
			cnt += 1

		prev = far

	# Determine the number of fingers held up. 
	gestures = ["One", "Two", "Three", "Four", "Five"]

	for i in range(len(gestures)):
		# This is a really cheeky way of evaluating for the hand gesture where 
		# there are zero fingers being held up.  
		# We determine zero fingers are being held up because the cnt is 1 and 
		# there are many convexity defects detected in the image from the 
		# knuckles of the hand being misaligned. 
		if (cnt == 0) and (cnt_used_defects > 10):
			# print(defects.shape[0])
			cv2.putText(img=orig_image, 
						text="Zero", 
						org=(5, 100), 
						fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
						fontScale=2, 
						color=(255,255,255), 
						thickness=8)

		# The real evaluation takes place here. 
		elif cnt == i:
			cv2.putText(img=orig_image, 
						text=gestures[i], 
						org=(5, 100), 
						fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
						fontScale=2, 
						color=(255,255,255), 
						thickness=8)

	return orig_image

def main():
	# Want to have the image path passed in command line. 
	# This will be testing images for now until time comes we can start testing
	# out on video. 
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--image", required=True, help="Path to image file.")

	args = vars(ap.parse_args())

	if not os.path.exists(args["image"]): 
		print("Bad image path...")
		return -1

	# Use the below process for putting a single image through the gesture 
	# recognizer.
	# ==========================================================================
	image = cv2.imread(args["image"], cv2.IMREAD_COLOR)

	image = gesture_recognizer(image)

	# Show and save our example. 
	cv2.imshow(args["image"], image)
	cv2.imwrite("processed_image.jpg", image)

	cv2.waitKey(0)
	cv2.destroyAllWindows()
	# ==========================================================================

	return 0


if __name__ == '__main__':
	main()
