from scipy.spatial import distance
from imutils import face_utils
import imutils
import dlib
import cv2
import pybind11

import winsound
import numpy as np

import time

import math

cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector() 
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") 




def eye_aspect_ratio(eye):
	A = distance.euclidean(eye[1], eye[5])
	B = distance.euclidean(eye[2], eye[4])
	C = distance.euclidean(eye[0], eye[3])
	ear = (A + B) / (2.0 * C)
	return ear
	#It defines the eye aspect ratio function and it calculates the distance
	#between eye landmarks and horizontal eye landmarks


	#if the eye is the eye aspect ratio will remain constant
	#but if it is closed, then the ratio will be way much smaller compare to when the eye is open

thresh = 0.25
frame_check = 20
	#2 constants were defined one for EAR to detect eyes closed and the other for the frames which the eye 
	#will trigger the alert

detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")# Dat file is the crux of the code

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
cap=cv2.VideoCapture(0)
flag=0
pTime=0

while True:



	ret, frame=cap.read()
	frame = imutils.resize(frame, width=450)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	subjects = detect(gray, 0)
	for subject in subjects:


		cTime=time.time()
		fps = 1/(cTime-pTime)
		pTime = cTime
		cv2.putText(frame, f'FPS: {int(fps)}',(20,70),cv2.FONT_HERSHEY_PLAIN,
		3,(255,0,0),1)

		#it helps shaping the facial landmarks and convert it to x,y cordinates through NumPy array
		shape = predict(gray, subject)
		shape = face_utils.shape_to_np(shape)#converting to NumPy Array
		#it takes both eyes cordinates to calculate the EAR for both eyes
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)
		#average EAR for both eyes (for 2 eyes)
		ear = (leftEAR + rightEAR) / 2.0
		#it calculates the convex hull for both eyes and visualize it using cv2 
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)



		if ear < thresh:
			#checking EAR is below the blink threshold
			#if true, increment blink frame counter
			flag += 1
			#if the eyes were closed for an amount of time it triggers the alert and sound alarm
			print (flag)
			if flag >= frame_check:

				cv2.putText(frame, "****************SLEEPY ALERT!****************", (10, 30),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
				cv2.putText(frame, "****************SLEEPY ALERT!****************", (10,325),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

				faces = detector(gray)
				for (i, face) in enumerate(faces):
					point = predictor(gray, face)
					points = face_utils.shape_to_np(point)

					for (x, y) in points :
						cv2.circle(frame, (x, y), 1, (0, 0, 255), 3)

				if cv2.waitKey(1) & 0xFF == ord('q'):
					break

				#print ("Drowsy")
		else:
			flag = 0



	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break





cv2.destroyAllWindows()
cap.release() 
