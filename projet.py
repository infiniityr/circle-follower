from collections import deque
from dotenv import load_dotenv
import numpy as np
import imutils
import cv2
import os

load_dotenv()

def getCirclesColor(img, color):
    blurred = cv2.GaussianBlur(img, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # construct a mask for the color, then perform
    # a series of dilations and erosions to remove any small
    # blobs left in the mask
    mask = cv2.inRange(hsv, np.array(eval(os.getenv(color + "_LIGHT_HSV"))), np.array(eval(os.getenv(color + "_DARK_HSV"))))
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # find contours in the mask and initialize the current
    # (x, y) center of the ball
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
    	cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    if os.getenv("SHOW_MASK") == "true":
        cv2.imshow('hsv_' + color, mask)

     # only proceed if at least one contour was found
    if len(cnts) > 0:
        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle and
        # centroid
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)

        # only proceed if the radius meets a minimum size
        if radius > eval(os.getenv("CIRCLE_MIN_RADIUS")):
            # Return the center of the circle and its radius
            return [[int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]), radius]]

    return None

# initialize the list of tracked points and the frame counter
colors = eval(os.getenv("SEARCH_COLOR"))
pts = [None] * len(colors)
for i in np.arange(0, len(colors)):
    pts[i] = deque(maxlen=eval(os.getenv("PATH_LENGTH")))
counter = 0

# Get the flux of the webcam
vs = cv2.VideoCapture(eval(os.getenv("INDEX_OF_DEVICE")))

while True:
    # grab the current frame
    ret, frame = vs.read()
 
    # resize the frame, blur it, and convert it to the HSV
    # color space
    frame = imutils.resize(frame, width=600)
    frame_cpy = frame.copy()

    # Iterates through the colors
    for index, col in enumerate(colors):
        circles = getCirclesColor(frame, col)
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles:
                # draw the outer circle
                cv2.circle(frame_cpy,(i[0],i[1]),i[2],eval(os.getenv(col + "_CIRCLE")),5)
                # draw the center of the circle
                cv2.circle(frame_cpy,(i[0],i[1]),2,(0,0,255),3)
                pts[index].appendleft((i[0],i[1]))

        if os.getenv("DRAW_PATH") == "true":
            # loop over the set of tracked points
            for i in np.arange(1, len(pts[index])):
                # if either of the tracked points are None, ignore them
                if pts[index][i - 1] is None or pts[index][i] is None:
                    continue
                # otherwise, compute the thickness of the line and
                # draw the connecting lines
                thickness = int(np.sqrt(eval(os.getenv("PATH_LENGTH")) / float(i + 1)) * 2.5)
                cv2.line(frame_cpy, pts[index][i - 1], pts[index][i], eval(os.getenv(col + "_PATH")), thickness)
 
    # show the frame to our screen and increment the frame counter
    counter += 1
 
    cv2.imshow('circles', frame_cpy)

    k = cv2.waitKey(20) & 0xFF
    if k == 27:
        break