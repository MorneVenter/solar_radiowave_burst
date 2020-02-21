
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
cv2.rectangle(mask, (320,142), (1216,408), (255), thickness = -1)

# closing
kernel = np.ones((15,15),np.uint8)
dilation = cv2.dilate(cv2.bitwise_and(blackAndWhiteImage, mask),kernel,iterations = 10)
erosion = cv2.erode(dilation,kernel,iterations = 10)

#smooth
kernel = np.ones((10,10),np.float32)/100
erosion = cv2.filter2D(erosion,-1,kernel)

# hough lines
# edges = cv2.Canny(erosion,50,150,apertureSize = 3)
minLineLength = 50
maxLineGap = 10
lines = cv2.HoughLinesP(erosion,1,np.pi/180,42,minLineLength,maxLineGap)
if not(lines is None):
    for x1,y1,x2,y2 in lines[0]:
        cv2.line(originalImage,(x1,y1),(x2,y2),(0,255,0),2)


#show img
cv2.imshow('Final Image', originalImage)
cv2.waitKey(0)

# cleanup
cv2.destroyAllWindows()
os.remove('pre-proc.png')
