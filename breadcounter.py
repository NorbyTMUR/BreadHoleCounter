# Standard imports
import cv2
import argparse
import os
import numpy as np;

'''parser = argparse.ArgumentParser(description='Template matcher')
parser.add_argument('--image', type=str, action='store',
                    help='The image to be used as template')
parser.add_argument('--show', action='store_true',
                    help='Shows result image')
parser.add_argument('--save-dir', type=str, default='./',
                    help='Directory in which you desire to save the result image')

args = parser.parse_args()'''

# Read image
i = cv2.imread("breadslice4.1.jpg", cv2.IMREAD_GRAYSCALE)
im = cv2.resize(i, None, fx=0.2, fy=0.2, interpolation=cv2.INTER_LINEAR)
'''ret, thresh = cv2.threshold(im, 127, 255, 0)
im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)'''

# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()

# Filter by Area.
params.filterByArea = True
params.minArea = 100
 
# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.57
 
# Filter by Inertia
params.filterByInertia = True
params.minInertiaRatio = 0.35

# Set up the detector with default parameters.
detector = cv2.SimpleBlobDetector_create(params)

# Detect blobs.
keypoints = detector.detect(im)
print("Matches:", len(keypoints))

# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
 
# Show keypoints
#cv2.imwrite(os.path.join(im, 'output.jpg'), im_with_keypoints)
cv2.imshow("Keypoints", im_with_keypoints)
cv2.waitKey(0)
cv2.destroyAllWindows()
