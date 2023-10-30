import cv2
import numpy as np
import matplotlib.pyplot as plt

def lowpassFilter(img, conv_kernel):
    row = img.shape[0]
    col = img.shape[1]

    kernel_size = conv_kernel.shape[0]
    half_kernel = np.int32(np.floor(kernel_size / 2))

    img_filtered = np.zeros((row, col), np.uint8)
    for y in range(row):
        for x in range(col):
            img_filtered[y, x] = 0
            for i in range(kernel_size):
                yy = y + i - half_kernel
                if (yy < 0) or (yy >= row - 1):
                    continue

                for j in range(kernel_size):
                    xx = x + j - half_kernel
                    if (xx < 0) or (xx >= col - 1):
                        continue

                    img_filtered[y, x] += img[yy, xx] * conv_kernel[i, j]

    img_filtered = np.uint8(np.floor(img_filtered))
    return img_filtered

def highpassFilter(img, lowpass):
    img_sharpened = img - lowpass
    return img_sharpened

def bandstopFilter(lowpass, highpass):
    img_filtered = lowpass + highpass
    return img_filtered

def bandpassFilter(img, bandstop):
    img_filtered = img - bandstop
    return img_filtered


img = cv2.imread('data/Tower2ITS.jpg', 0)

kernel_size = 3
conv_kernel_1 = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
conv_kernel_2 = np.ones((kernel_size + 1, kernel_size + 1), np.float32) / (kernel_size + 1 * kernel_size + 1)

img_blurred_1 = lowpassFilter(img, conv_kernel_1)
img_blurred_2 = lowpassFilter(img, conv_kernel_2)

img_sharpened_1 = highpassFilter(img, img_blurred_1)
img_sharpened_2 = highpassFilter(img, img_blurred_2)

img_bandstop = bandstopFilter(img_blurred_1, img_sharpened_2)

img_bandpass = bandpassFilter(img, img_bandstop)

# cv2.imshow('Original', img)
# cv2.imshow('Lowpass', img_blurred_1)
# cv2.imshow('Highpass', img_sharpened_1)
# cv2.imshow('Bandstop', img_bandstop)
# cv2.imshow('Bandpass', img_bandpass)

combined = np.vstack((np.hstack((img, img_blurred_1)), np.hstack((img_bandpass, img_sharpened_1))))
cv2.imshow('Combined', combined)
cv2.imwrite('data/Combined.jpg', combined)


cv2.waitKey(0)
cv2.destroyAllWindows()