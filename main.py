# import the necessary packages
from __future__ import print_function
from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
import takepicture
import os
import time
import urllib2
import json

def analyze(imagePath):
	# load the image and resize it to (1) reduce detection time
	# and (2) improve detection accuracy
	image = cv2.imread(imagePath)
	image = imutils.resize(image, width=min(400, image.shape[1]))
	orig = image.copy()

	# detect people in the image
	(rects, weights) = hog.detectMultiScale(image, winStride=(4, 4),
		padding=(8, 8), scale=1.05)

	# draw the original bounding boxes
	for (x, y, w, h) in rects:
		cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)

	# apply non-maxima suppression to the bounding boxes using a
	# fairly large overlap threshold to try to maintain overlapping
	# boxes that are still people
	rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
	pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

	# draw the final bounding boxes
	for (xA, yA, xB, yB) in pick:
		cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)

	# show some information on the number of bounding boxes
	filename = imagePath[imagePath.rfind("/") + 1:]
	print("[INFO] {}: {} original boxes, {} after suppression".format(
		filename, len(rects), len(pick)))

	# show the output images
	cv2.imshow("Before NMS", orig)
	cv2.imshow("After NMS", image)
	
class Point(object):
	x = 0.0
	y = 0.0
	
	def __init__(self, x, y):
		self.x = x
		self.y = y
		
	def __str__(self):
		return str(self.x) + ", " + str(self.y)
	
def getPoint(boundingBox):
	return Point(
		boundingBox["left"] + boundingBox["width"] / 2,
		boundingBox["top"] + boundingBox["height"] / 2)
	
def analyzeAzure(imagePath):
	with open(imagePath, 'r') as imageFile:
		imageData = imageFile.read()
		
		url = "https://southcentralus.api.cognitive.microsoft.com/customvision/v2.0/Prediction/3c8ab5ac-1b61-4040-826f-d0ce69d5ecb7/image?iterationId=91d3f3ae-17e1-4bc6-bb2a-eabfaf757622"
		#headers = "Prediction-Key=161975688d084e45bcfc2a07a8a35239&Content-Type=application/octet-stream"
		headers = {
			"Prediction-Key": "161975688d084e45bcfc2a07a8a35239",
			"Content-Type": "application/octet-stream"
		}
		req = urllib2.Request(url, imageData, headers)
		resp = urllib2.urlopen(req)
		respObj = json.loads(resp.read());
		
		predictions = respObj["predictions"];
		people = 0
		points = []
		for prediction in predictions:
			if prediction["probability"] >= 0.5:
				people += 1
				point = getPoint(prediction["boundingBox"])
				points.append(point)
				print(str(point))
		
		print("People: " + str(people))


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=False, help="path to images directory")
args = vars(ap.parse_args())

# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

if args["images"]:
	# loop over the image paths
	for imagePath in paths.list_images(args["images"]):
		analyze(imagePath)
		cv2.waitKey(0)
else:
	while True:
		# take a new picture
		takepicture.take_picture("./tmp/pic.png")
		analyzeAzure(os.path.abspath("./tmp/pic.png"))
		#cv2.waitKey(8)
		time.sleep(3)



