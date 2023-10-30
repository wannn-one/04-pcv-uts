import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('data/Tower2ITS.jpg', 0)

r1 = 0
s1 = 40
r2 = 250
s2 = 255

def T(x, r1, s1, r2, s2):
    s = 0 
    if (0 < x) and (x < r1):
        s = s1 / r1 * x 
    elif (r1 <= x) and (x < r2): 
        s = (s2 - s1) / (r2 - r1) * (x - r1) + s1 
    elif (r2 <= x) and (x <= 255) and (r2 < 255): 
        s = (255 - s2) / (255 - r2) * (x - r2) + s2 
    else: 
        s = s2
    s = np.uint8(np.floor(s))
    return s

b = img.shape[0]
k = img.shape[1]
im = np.zeros((b, k), np.uint8)

for i in range(b):
    for j in range(k):
        x = img[i, j]
        im[i, j] = T(x, r1, s1, r2, s2) 

histogram_img = cv2.calcHist([img], [0], None, [256], [0, 256])
histogram_im = cv2.calcHist([im], [0], None, [256], [0, 256])

plt.figure(figsize=(8, 8))

plt.subplot(2, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Citra Asli')
plt.subplot(2, 2, 2)
plt.bar(np.arange(0, 256, 1), histogram_img[:, 0])
plt.title('Original')

plt.subplot(2, 2, 3)
plt.imshow(im, cmap='gray')
plt.title('Citra Kontrast')
plt.subplot(2, 2, 4)
plt.bar(np.arange(0, 256, 1), histogram_im[:, 0])
plt.title('Contrast Result')

plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
