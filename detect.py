
import matplotlib.pyplot as plt
from radiospectra.sources.callisto import CallistoSpectrogram
import numpy as np
import cv2
import os
import math

#load all files
rootdir = 'data/'
extensions = ('.fit.gz')
datafiles = []
for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        ext = os.path.splitext(file)[-1].lower()
        if ext in extensions:
            datafiles.append(os.path.join(subdir, file))

for file in datafiles:
    # load FIT data
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

    # load and convert to grayscale
    outImage = cv2.imread('output.png')
    originalImage = cv2.imread('pre-proc.png')
    grayImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)

    #median filter
    kernel_med = np.ones((6,6),np.uint8)/30
    med = cv2.filter2D(grayImage,-1,kernel_med)

    # convert to binary img
    (thresh, blackAndWhiteImage) = cv2.threshold(med, 90, 255, cv2.THRESH_BINARY)

    # mask spectogram
    mask = np.zeros(blackAndWhiteImage.shape[:2], dtype=np.uint8)
    #cv2.rectangle(mask, (320,142), (1216,408), (255), thickness = -1)
    cv2.rectangle(mask, (324,76), (1215,479), (255), thickness = -1)
    crop = cv2.bitwise_and(blackAndWhiteImage, mask)

    # Erosion
    e_kernel = np.ones((3,3),np.uint8)
    erosion = cv2.erode(crop,e_kernel,iterations = 2)

    # med 2
    kernel_med = np.ones((6,6),np.uint8)/36
    final = cv2.filter2D(erosion,-1,kernel_med)

    # # closing
    # kernel = np.ones((15,15),np.uint8)
    #
    # dilation = cv2.dilate(crop,kernel,iterations = 10)
    # erosion = cv2.erode(dilation,kernel,iterations = 10)
    #
    # # smooth
    # kernel = np.ones((10,10),np.float32)/100
    # smooth = cv2.filter2D(erosion,-1,kernel)

    # hough lines
    minLineLength = 50
    maxLineGap = 40
    rho = 3
    theta = np.pi/180
    threshold = 250
    lines = cv2.HoughLinesP(final,rho,theta,threshold,minLineLength,maxLineGap)
    if not(lines is None):
        ln = np.array([100000,0,0,100000])
        for line in lines:
            for x1,y1,x2,y2 in line:
                slope = (y2-y1)/(x2-x1)
                if slope<0 and slope>-math.inf and abs(slope)>0.06:
                    if x1 < ln[0]:ln[0] = x1
                    if y1 > ln[1]:ln[1] = y1
                    if x2 < ln[0]:ln[0] = x2
                    if y2 > ln[1]:ln[1] = y2

                    if x1 > ln[2]:ln[2] = x1
                    if y1 < ln[3]:ln[3] = y1
                    if x2 > ln[2]:ln[2] = x2
                    if y2 < ln[3]:ln[3] = y2

        if not(np.array_equal(ln,[100000,0,0,100000])):
            final_slope = (ln[3]-ln[1])/(ln[2]-ln[0])
            cv2.line(outImage,(ln[0],ln[1]),(ln[2],ln[3]),(255,255,255),2)
            print(final_slope)
            if final_slope <= -3:
                cv2.putText(outImage, 'Type III', (ln[0]+20,ln[3]), cv2.FONT_HERSHEY_COMPLEX , 0.7,(255,255, 255), 1, cv2.LINE_AA)
            elif final_slope > -3:
                cv2.putText(outImage, 'Type II', (ln[0]+20,ln[3]), cv2.FONT_HERSHEY_COMPLEX , 0.7,(255,255, 255), 1, cv2.LINE_AA)


    # show images, uncomment to see process
    # cv2.imshow('Gray Image    ', grayImage)
    # cv2.imshow('Binary Image', blackAndWhiteImage)
    # cv2.imshow('Cropped Image', crop)
    # cv2.imshow('Erosion Image', erosion)
    cv2.imshow('Smooth Image', final)
    # cv2.imshow('Final Image', originalImage)

    # show pectrogram
    cv2.imshow('Spectrogram', outImage)

    cv2.waitKey(0)

    # cleanup
    cv2.destroyAllWindows()
    os.remove('pre-proc.png')
    os.remove('output.png')
