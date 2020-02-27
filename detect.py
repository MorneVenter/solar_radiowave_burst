
import matplotlib.pyplot as plt
from radiospectra.sources.callisto import CallistoSpectrogram
import numpy as np
import cv2
import os

# load FIT data
file = './data/type3.fit.gz'
image = CallistoSpectrogram.read(file)
nobg = plt.figure(figsize=(16,6))
nobg = image.subtract_bg()

# plots spectogram and saves img
nobg.plot(vmin=12, vmax = 255, cmap='inferno')
#plt.axis('equal')
plt.title ('Result')
plt.savefig('pre-proc.png')
image.plot()
plt.savefig('output.png')

# convert to binary img
outImage = cv2.imread('output.png')
originalImage = cv2.imread('pre-proc.png')
grayImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)
(thresh, blackAndWhiteImage) = cv2.threshold(grayImage, 115, 255, cv2.THRESH_BINARY)

# mask spectogram
mask = np.zeros(blackAndWhiteImage.shape[:2], dtype=np.uint8)
#cv2.rectangle(mask, (320,142), (1216,408), (255), thickness = -1)
cv2.rectangle(mask, (324,76), (1216,480), (255), thickness = -1)

# closing
kernel = np.ones((15,15),np.uint8)
crop = cv2.bitwise_and(blackAndWhiteImage, mask)
dilation = cv2.dilate(crop,kernel,iterations = 10)
erosion = cv2.erode(dilation,kernel,iterations = 10)

# smooth
kernel = np.ones((10,10),np.float32)/100
smooth = cv2.filter2D(erosion,-1,kernel)

# hough lines
minLineLength = 50
maxLineGap = 100
lines = cv2.HoughLinesP(smooth,5,3*np.pi/180,42,minLineLength,maxLineGap)
if not(lines is None):
    for x1,y1,x2,y2 in lines[0]:
        slope = abs((y2-y1)/(x2-x1))
        if not(slope == 0):
            cv2.line(originalImage,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.line(outImage,(x1,y1),(x2,y2),(0,0,255),2)
            print(slope)
            if slope >= 3:
                cv2.putText(outImage, 'Type III', (x1+20,y2), cv2.FONT_HERSHEY_COMPLEX , 0.7,(0,0, 255), 1, cv2.LINE_AA)
            elif slope < 3:
                cv2.putText(outImage, 'Type II', (x1+20,y2), cv2.FONT_HERSHEY_COMPLEX , 0.7,(0,0, 255), 1, cv2.LINE_AA)


# show images, uncomment to see process
# cv2.imshow('Gray Image', grayImage)
# cv2.imshow('Binary Image', blackAndWhiteImage)
# cv2.imshow('Cropped Image', crop)
# cv2.imshow('Erosion Image', erosion)
# cv2.imshow('Smooth Image', smooth)
# cv2.imshow('Final Image', originalImage)

# show pectrogram
cv2.imshow('Spectrogram', outImage)

cv2.waitKey(0)

# cleanup
cv2.destroyAllWindows()
os.remove('pre-proc.png')
os.remove('output.png')
