
import matplotlib.pyplot as plt
from radiospectra.sources.callisto import CallistoSpectrogram
import numpy as np
import cv2
import math
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

#Blob detection transform
surf = cv2.xfeatures2d.SURF_create(35000) #create the SURF object

if not(surf.getUpright()):
    surf.setUpright(True)
if not(surf.getExtended()):
    surf.setExtended(True)

kp, des = surf.detectAndCompute(blackAndWhiteImage, mask) #find keypoints and descriptors
print('Size of descriptors: ' + str(surf.descriptorSize()))
print('Number of keypoints: ' + str(len(kp)))
final_img = cv2.drawKeypoints(blackAndWhiteImage,kp,None,(255,0,0),4) # draws keypoints to image
#for point in kp:
    #print('Blob at:' + str(round(point.pt[0],2)) +' ' + str(round(point.pt[1],2)))

#convert keypionts to numpy array
npar = np.empty([len(kp), 2])
i = 0
for point in kp:
    npar[i] = (point.pt[0], point.pt[1])
    i=i+1
#print(npar)

#draw line on image
img_line = cv2.polylines(originalImage, np.int32([npar]), isClosed = False,color = (255,255,255),thickness = 2)

#show img
cv2.imshow('Final image', img_line)
cv2.waitKey(0)

# cleanup
cv2.destroyAllWindows()
os.remove('pre-proc.png')
