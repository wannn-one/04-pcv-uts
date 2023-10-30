import cv2
import numpy as np

img = cv2.imread('data/Tower2ITS.jpg', 0)

sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])

sobel_y = np.array([[-1, -2, -1],
                    [0, 0, 0],
                    [1, 2, 1]])

# filter2D(src, ddepth, kernel)
sobel_x = cv2.filter2D(img, -5, sobel_x)
sobel_y = cv2.filter2D(img, -5, sobel_y)

# display images
combined = np.vstack((np.hstack((img, sobel_x)), np.hstack((sobel_y, sobel_x + sobel_y))))
cv2.imshow('Sobel', combined)
cv2.imwrite('data/Sobel.jpg', combined)

cv2.waitKey(0)
cv2.destroyAllWindows()