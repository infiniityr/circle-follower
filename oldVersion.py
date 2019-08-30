import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors
from dotenv import load_dotenv

def getCirclesColor(img, color):
    frame_gau_blur = cv2.GaussianBlur(frame, (21, 21), 0)
    # converting BGR to HSV
    hsv = cv2.cvtColor(frame_gau_blur, cv2.COLOR_RGB2HSV)

    color_lower = np.array(eval(os.getenv(color + "_LIGHT_HSV")))
    color_higher = np.array(eval(os.getenv(color + "_DARK_HSV")))

    color_range = cv2.inRange(hsv, color_lower, color_higher)

    res_color = cv2.bitwise_and(frame_gau_blur,frame_gau_blur, mask=color_range)

    cv2.imshow('shape_' + color, res_color)

    color_s_gray = cv2.cvtColor(res_color, cv2.COLOR_BGR2GRAY)
    canny_edge = cv2.Canny(color_s_gray, 50, 240)
    # applying HoughCircles
    return cv2.HoughCircles(canny_edge, cv2.HOUGH_GRADIENT,
                                dp=int(os.getenv("CIRCLE_DP")), 
                                minDist=int(os.getenv("CIRCLE_MIN_DIST")),
                                param1=int(os.getenv("CIRCLE_PARAM1")),
                                param2=int(os.getenv("CIRCLE_PARAM2")),
                                minRadius=int(os.getenv("CIRCLE_MIN_RADIUS")),
                                maxRadius=int(os.getenv("CIRCLE_MAX_RADIUS")))

load_dotenv()

cap = cv2.VideoCapture(int(os.getenv("INDEX_OF_DEVICE")))

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame_cpy = frame.copy()

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if os.getenv("RGB_COLOR_GRAPH") == "true":
        r, g, b = cv2.split(frame)
        fig = plt.figure()
        axis = fig.add_subplot(1, 1, 1, projection="3d")

        pixel_colors = frame.reshape((np.shape(frame)[0]*np.shape(frame)[1], 3))
        norm = colors.Normalize(vmin=-1.,vmax=1.)
        norm.autoscale(pixel_colors)
        pixel_colors = norm(pixel_colors).tolist()

        axis.scatter(r.flatten(), g.flatten(), b.flatten(), facecolors=pixel_colors, marker=".")
        axis.set_xlabel("Red")
        axis.set_ylabel("Green")
        axis.set_zlabel("Blue")
        plt.show()

    if os.getenv("HSV_COLOR_GRAPH") == "true":
        frame_gau_blur = cv2.GaussianBlur(frame, (21, 21), 0)
        # converting BGR to HSV
        hsv = cv2.cvtColor(frame_gau_blur, cv2.COLOR_RGB2HSV)

        h, s, v = cv2.split(hsv)
        fig = plt.figure()
        axis = fig.add_subplot(1, 1, 1, projection="3d")

        pixel_colors = img.reshape((np.shape(img)[0]*np.shape(img)[1], 3))
        norm = colors.Normalize(vmin=-1.,vmax=1.)
        norm.autoscale(pixel_colors)
        pixel_colors = norm(pixel_colors).tolist()

        axis.scatter(h.flatten(), s.flatten(), v.flatten(), facecolors=pixel_colors, marker=".")
        axis.set_xlabel("Hue")
        axis.set_ylabel("Saturation")
        axis.set_zlabel("Value")
        plt.show()

    for col in eval(os.getenv("SEARCH_COLOR")):
        circles = getCirclesColor(frame, col)

        if circles is not None:
            circles = np.uint16(np.around(circles))

            for i in circles[0,:]:
                # draw the outer circle
                cv2.circle(frame_cpy,(i[0],i[1]),i[2],eval(os.getenv(col + "_CIRCLE")),5)
                # draw the center of the circle
                cv2.circle(frame_cpy,(i[0],i[1]),2,(0,0,255),3)


    cv2.imshow('circles', frame_cpy)

    k = cv2.waitKey(20) & 0xFF
    if k == 27:
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()