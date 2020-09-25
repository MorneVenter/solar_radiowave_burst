
import matplotlib.pyplot as plt
from radiospectra.sources.callisto import CallistoSpectrogram
import numpy as np
import cv2
import os
import math
from scipy.spatial import distance
import itertools
from imutils import build_montages
from imutils import paths
import intersect as IS
import argparse

debug = False
print("-------------------------------------------------------------------------")
print("             ,")
print("            /|      __")
print("           / |   ,-~ /")
print("          Y :|  //  /")
print("          | jj /( .^")
print("          >-\"~\"-v\"")
print("         /       Y")
print("        j>  <    |")
print("       ( ~T~     j")
print("        >._-' _./")
print("       /   \"~\"  |")
print("      Y     _,  |")
print("     /| ;-\"~ _  l")
print("    / l/ ,-\"~    \"")
print("    \//\/      .- \\")
print("     Y        /    Y    -Let's go!")
print("     l       I     !")
print("     ]\      _\    /\"")
print("    (\" ~----( ~   Y.\)")
print("~~~~~~~~~~~~~~~~~~~~~~~~~")
print("-------------------------------------------------------------------------")
print("DEVELOPED BY MORNE VENTER")
print("-------------------------------------------------------------------------")
print("TO RUN IN DEBUG MODE TYPE \"python3 detect.py -d\"")
print("Place your .FITS file in the DATA folder.")
print("You will find your results in the DETECTED_SRB folder.")
print("Starting  SRB detection system ... ")
print("-------------------------------------------------------------------------")


#use -d tp show debug info
parser = argparse.ArgumentParser(description='Enabled debugging.')
parser.add_argument("-d", "--debug", help="Enabled debugging options.", action='store_true')
args = parser.parse_args()
dbg = args.debug

if isinstance(dbg, bool) and dbg==True:
    debug = True

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
    spectrogram_name = str(image.get_header()['DATE']) + "_" + str(image.get_header()['TIME-OBS'])

    nobg = plt.figure(figsize=(16,6))
    nobg = image.subtract_bg()

    # plots spectogram and saves img
    nobg.plot(vmin=0, vmax = 200, cmap='inferno')
    #plt.axis('equal')
    plt.title ('Result')
    plt.savefig('pre-proc.png')
    image.plot()
    plt.savefig('output.png')

    # load and convert to grayscale
    outImage = cv2.imread('output.png')
    originalImage = cv2.imread('pre-proc.png')
    grayImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)

    # mask spectogram
    mask = np.zeros(grayImage.shape[:2], dtype=np.uint8)
    cv2.rectangle(mask, (324,76), (1215,479), (255), thickness = -1)
    crop = cv2.bitwise_and(grayImage, mask)

    #avg filter
    kernel_med = np.ones((6,6),np.uint8)/30
    med = cv2.filter2D(crop,-1,kernel_med)


    # convert to binary img
    (thresh, blackAndWhiteImage) = cv2.threshold(med, 45, 255, cv2.THRESH_BINARY) #was 90


    # Erosion
    e_kernel = np.ones((3,3),np.uint8)
    erosion = cv2.erode(blackAndWhiteImage,e_kernel,iterations = 1)

    # avg 2
    kernel_med = np.ones((6,6),np.uint8)/36
    final = cv2.filter2D(erosion,-1,kernel_med)

    burst_found = False

    # hough lines
    minLineLength = 130
    maxLineGap = 1
    rho = 4 #was 3
    theta = np.pi/180
    threshold = 300
    lines = cv2.HoughLinesP(final,rho,theta,threshold,minLineLength,maxLineGap)
    if not(lines is None):
        valid_lines = []
        for line in lines:
            for x1,y1,x2,y2 in line:
                if x2-x1 == 0:
                    slope = math.inf
                else:
                    slope = (y2-y1)/(x2-x1)
                if slope<0 and slope>-math.inf and abs(slope)>0.1:
                    valid_lines.append([x1,y1,x2,y2])
                    if debug:
                        cv2.line(outImage,(x1,y1),(x2,y2),(255,255,0),2)
        neighbor_dist = []
        for ln in valid_lines:
            near = []
            r = 12
            if debug:
                cv2.circle(outImage,(ln[0],ln[1]),int(r),(255,255,0),1,8,0)
                cv2.circle(outImage,(ln[2],ln[3]),int(r),(255,255,0),1,8,0)
            near.append([ln[0],ln[1]])
            near.append([ln[2],ln[3]])
            for ln2 in valid_lines:
                if distance.euclidean((ln[0],ln[1]), (ln2[0],ln2[1])) < r*2:
                    near.append([ln2[0],ln2[1]])
                if distance.euclidean((ln[0],ln[1]), (ln2[2],ln2[3])) < r*2:
                    near.append([ln2[2],ln2[3]])
                if distance.euclidean((ln[2],ln[3]), (ln2[0],ln2[1])) < r*2:
                    near.append([ln2[0],ln2[1]])
                if distance.euclidean((ln[2],ln[3]), (ln2[2],ln2[3])) < r*2:
                    near.append([ln2[2],ln2[3]])
            if not(near == []):
                neighbor_dist.append(near)

        for i in range(0,len(neighbor_dist)-1):
            for k in range(i+1, len(neighbor_dist)):
                for pt in neighbor_dist[i]:
                    if(pt in neighbor_dist[k]):
                        neighbor_dist[i] = neighbor_dist[i] + neighbor_dist[k]
                        neighbor_dist[i] = list(k for k,_ in itertools.groupby(neighbor_dist[i]))
                        neighbor_dist[k]=[]
        final_lines=[]
        for row in neighbor_dist:
            ln = np.array([100000,0,0,100000])
            for pt in row:
                if pt[0] < ln[0]:ln[0] = pt[0]
                if pt[1] > ln[1]:ln[1] = pt[1]
                if pt[0] > ln[2]:ln[2] = pt[0]
                if pt[1] < ln[3]:ln[3] = pt[1]

            if not(np.array_equal(ln,[100000,0,0,100000])):
                final_lines.append(ln);

        for ln1 in range(0, len(final_lines)-1):
            for ln2 in range(ln1+1, len(final_lines)):
                pt1 = IS.Point(final_lines[ln1][0],final_lines[ln1][1])
                pt2 = IS.Point(final_lines[ln1][2],final_lines[ln1][3])
                pt3 = IS.Point(final_lines[ln2][0],final_lines[ln2][1])
                pt4 = IS.Point(final_lines[ln2][2],final_lines[ln2][3])
                if IS.doIntersect(pt1, pt2, pt3, pt4):
                    l1Len = IS.getLength(pt1, pt2)
                    l2Len = IS.getLength(pt3, pt4)
                    if l1Len < l2Len:
                        final_lines[ln1] = [0,0,0,0]
                    else:
                        final_lines[ln2] = [0,0,0,0]

        if len(final_lines) > 0:
            ln = final_lines[0]
            for line_f in final_lines:
                if not(np.array_equal(ln,[0,0,0,0])):
                    len1 = math.sqrt((ln[3]-ln[1])**2 + (ln[2]-ln[0])**2)
                    len2 = math.sqrt((line_f[3]-line_f[1])**2 + (line_f[2]-line_f[0])**2)
                    if len2 > len1:
                        ln = line_f
            if math.sqrt((ln[3]-ln[1])**2 + (ln[2]-ln[0])**2) > 50.0:
                final_slope = (ln[3]-ln[1])/(ln[2]-ln[0])
                print(final_slope)
                if not(final_slope > -0.1):
                    if debug:
                        cv2.line(outImage,(ln[0],ln[1]),(ln[2],ln[3]),(255,0,255),2)
                    r = distance.euclidean((ln[0],ln[1]),(ln[2],ln[3]))
                    cv2.line(outImage,(ln[0],ln[1]),(ln[2],ln[3]),(255,255,255),3)
                    cv2.line(outImage,(ln[0],ln[1]),(ln[2],ln[3]),(0,0,255),2)
                    if final_slope <= -1.0:
                        cv2.putText(outImage, 'Type III', (ln[0]+20,ln[1]+20), cv2.FONT_HERSHEY_COMPLEX , 0.7,(255,255, 255), 2, cv2.LINE_AA)
                        cv2.putText(outImage, 'Type III', (ln[0]+20,ln[1]+20), cv2.FONT_HERSHEY_COMPLEX , 0.7,(0,0,255), 1, cv2.LINE_AA)
                        print("Type III SRB found at %s, saving results ... " % spectrogram_name)
                        burst_found = True
                    elif final_slope > -1.0:
                        cv2.putText(outImage, 'Type II', (ln[0]+20,ln[1]+20), cv2.FONT_HERSHEY_COMPLEX , 0.7,(255, 255, 255), 2, cv2.LINE_AA)
                        cv2.putText(outImage, 'Type II', (ln[0]+20,ln[1]+20), cv2.FONT_HERSHEY_COMPLEX , 0.7,(0,0,255), 1, cv2.LINE_AA)
                        print("Type II SRB found at %s, saving results ... " % spectrogram_name)
                        burst_found = True


    # show images
    if debug: #cvCvtColor(input, CV_GRAY2BGR)
        images = [
                    cv2.cvtColor(grayImage, cv2.COLOR_GRAY2RGB),
                    cv2.cvtColor(med, cv2.COLOR_GRAY2RGB),
                    cv2.cvtColor(blackAndWhiteImage, cv2.COLOR_GRAY2RGB),
                    cv2.cvtColor(erosion, cv2.COLOR_GRAY2RGB),
                    cv2.cvtColor(final, cv2.COLOR_GRAY2RGB),
                    outImage
                    ]
        montages = build_montages(images, (1600, 600), (2, 3))
        for montage in montages:
            if burst_found:
                cv2.imwrite('detected_SRB/%s.png' % str(spectrogram_name), montage)
            cv2.namedWindow("debug", cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty("debug",cv2.WINDOW_NORMAL,cv2.WINDOW_KEEPRATIO)
            cv2.imshow("debug", montage)

    # show pectrogram
    else:
        if burst_found:
            cv2.imwrite('detected_SRB/%s.png' % str(spectrogram_name), outImage)
        cv2.namedWindow("Final", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("Final",cv2.WINDOW_NORMAL,cv2.WINDOW_KEEPRATIO)
        cv2.imshow("Final", outImage)


    cv2.waitKey(0)   # wait for 1.2 second

    # cleanup
    cv2.destroyAllWindows()
    os.remove('pre-proc.png')
    os.remove('output.png')
