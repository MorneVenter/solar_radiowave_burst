
import matplotlib.pyplot as plt
from radiospectra.sources.callisto import CallistoSpectrogram
import numpy as np
import cv2
import os
import math
from scipy.spatial import distance
import itertools

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
    cv2.rectangle(mask, (324,76), (1215,479), (255), thickness = -1)
    crop = cv2.bitwise_and(blackAndWhiteImage, mask)

    # Erosion
    e_kernel = np.ones((3,3),np.uint8)
    erosion = cv2.erode(crop,e_kernel,iterations = 2)

    # med 2
    kernel_med = np.ones((6,6),np.uint8)/36
    final = cv2.filter2D(erosion,-1,kernel_med)


    # hough lines
    minLineLength = 50
    maxLineGap = 40
    rho = 3
    theta = np.pi/180
    threshold = 250
    lines = cv2.HoughLinesP(final,rho,theta,threshold,minLineLength,maxLineGap)
    if not(lines is None):
        valid_lines = []
        for line in lines:
            for x1,y1,x2,y2 in line:
                slope = (y2-y1)/(x2-x1)
                if slope<0 and slope>-math.inf and abs(slope)>0.06:
                    valid_lines.append([x1,y1,x2,y2])
                    #cv2.line(outImage,(x1,y1),(x2,y2),(0,0,255),2)
        neighbor_dist = []
        for ln in valid_lines:
            near = []
            r = distance.euclidean((ln[0],ln[1]), (ln[2],ln[3]))
            #cv2.circle(outImage,(ln[0],ln[1]),int(r),(0,0,255),1,8,0)
            #cv2.circle(outImage,(ln[2],ln[3]),int(r),(0,0,255),1,8,0)
            for ln2 in valid_lines:
                if distance.euclidean((ln[0],ln[1]), (ln2[0],ln2[1])) < r:
                    near.append([ln2[0],ln2[1]])
                if distance.euclidean((ln[0],ln[1]), (ln2[2],ln2[3])) < r:
                    near.append([ln2[2],ln2[3]])
                if distance.euclidean((ln[2],ln[3]), (ln2[0],ln2[1])) < r:
                    near.append([ln2[0],ln2[1]])
                if distance.euclidean((ln[2],ln[3]), (ln2[2],ln2[3])) < r:
                    near.append([ln2[2],ln2[3]])
            if not(near == []):
                neighbor_dist.append(near)

        for i in range(0,len(neighbor_dist)-1):
            for k in range(i+1, len(neighbor_dist)):
                for pt in neighbor_dist[i]:
                    if(pt in neighbor_dist[k]):
                        neighbor_dist[i] = neighbor_dist[i] + neighbor_dist[k]
                        neighbor_dist[i] = list(k for k,_ in itertools.groupby(neighbor_dist[i]))
                        #neighbor_dist[i] = neighbor_dist[i] + list(set(neighbor_dist[k]) - set(neighbor_dist[i]))
                        neighbor_dist[k]=[]

        for row in neighbor_dist:
            ln = np.array([100000,0,0,100000])
            for pt in row:
                if pt[0] < ln[0]:ln[0] = pt[0]
                if pt[1] > ln[1]:ln[1] = pt[1]
                if pt[0] > ln[2]:ln[2] = pt[0]
                if pt[1] < ln[3]:ln[3] = pt[1]

            if not(np.array_equal(ln,[100000,0,0,100000])):
                final_slope = (ln[3]-ln[1])/(ln[2]-ln[0])
                #cv2.line(outImage,(ln[0],ln[1]),(ln[2],ln[3]),(255,255,255),2)
                r = distance.euclidean((ln[0],ln[1]),(ln[2],ln[3]))
                cv2.circle(outImage,(int((ln[0]+ln[2])/2),int((ln[1]+ln[3])/2)),int(r/2),(255,255,255),2,8,0)
                print(final_slope)
                if final_slope <= -3:
                    cv2.putText(outImage, 'Type III', (ln[0]+20,ln[3]), cv2.FONT_HERSHEY_COMPLEX , 0.7,(255,255, 255), 1, cv2.LINE_AA)
                elif final_slope > -3:
                    cv2.putText(outImage, 'Type II', (ln[0]+20,ln[3]), cv2.FONT_HERSHEY_COMPLEX , 0.7,(255,255, 255), 1, cv2.LINE_AA)


    # show images, uncomment to see process
    cv2.imshow('Smooth Image', final)

    # show pectrogram
    cv2.imshow('Spectrogram', outImage)

    cv2.waitKey(0)

    # cleanup
    cv2.destroyAllWindows()
    os.remove('pre-proc.png')
    os.remove('output.png')
