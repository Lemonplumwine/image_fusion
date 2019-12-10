import numpy as np
import cv2
import matplotlib.pyplot as plt

src = cv2.imread('a.png')
sift = cv2.xfeatures2d.SIFT_create(0, 3, 0.04, 10)
key_points_for_all = []
descriptor_for_all = []
colors_for_all = []
key_points, descriptor = sift.detectAndCompute(cv2.cvtColor(src, cv2.COLOR_BGR2GRAY), None)
key_points_for_all.append(key_points)
descriptor_for_all.append(descriptor)
colors = np.zeros((len(key_points), 3))

for i, key_point in enumerate(key_points):
            p = key_point.pt
            colors[i] = src[int(p[1])][int(p[0])]
colors_for_all.append(colors)

img = cv2.drawKeypoints(src, key_points, src, color = (255, 0, 0))
cv2.namedWindow('Keypoint')
cv2.imshow('Keypoint', img)

k = cv2.waitKey(0)
if k == 0:
    cv2.destroyAllWindows()
elif k == ord('s'):
    cv2.destroyAllWindows()


