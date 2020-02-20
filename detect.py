
import matplotlib.pyplot as plt
from radiospectra.sources.callisto import CallistoSpectrogram
import numpy as np
import cv2
import os

#load FIT data
files = './data/type3.fit.gz'
image = CallistoSpectrogram.read(files)
nobg = plt.figure(figsize=(16,6))
nobg = image.subtract_bg()

#plots spectogram and saves img
nobg.plot(vmin=12, vmax = 255, cmap='inferno')
plt.axis('equal')
plt.title ('Type 3')
plt.savefig('pre-proc.png')

#convert to binary img
originalImage = cv2.imread('pre-proc.png')
grayImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)
(thresh, blackAndWhiteImage) = cv2.threshold(grayImage, 115, 255, cv2.THRESH_BINARY)

# mask spectogram
mask = np.zeros(blackAndWhiteImage.shape[:2], dtype=np.uint8)
cv2.rectangle(mask, (323,76), (1217,480), (255), thickness = -1)

# erosion
kernel = np.ones((5,5),np.uint8)
erosion = cv2.erode(cv2.bitwise_and(blackAndWhiteImage, mask),kernel,iterations = 1)


#show img
cv2.imshow('PreProc Image', erosion)
cv2.waitKey(0)

# cleanup
cv2.destroyAllWindows()
os.remove('pre-proc.png')
