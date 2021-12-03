#!/bin/python
import sys
import numpy as np
import cv2
from random import randint as rand
#from matplotlib import pyplot as plt

# liczba znakow 
TYPES   = 1

correct_moments = []

# poszukiwane ustawienia
HUE_MIN     = 190
HUE_MAX     = 220
AREA        = 0.5
MOMENTS     = []
THRESHOLD   = 42

def normalizeHist(image):
    (B, G, R) = cv2.split(image)
    B = cv2.equalizeHist(B)
    G = cv2.equalizeHist(G)
    R = cv2.equalizeHist(R)
    return cv2.merge((B, G, R))
# dziala?
def LaplacianOfGaussian(image):
    blur    = cv2.GaussianBlur(image, (3,3), 0)
    gray    = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    laplac  = cv2.Laplacian(gray, cv2.CV_8U,3,3,2)
    laplac  = cv2.convertScaleAbs(laplac)
    return laplac 
    
def thresholding(image, hue_min, hue_max):
    laplac = LaplacianOfGaussian(image)
    thres = cv2.threshold(laplac, THRESHOLD, 255, cv2.THRESH_BINARY)[1]
    #thres = cv2.bitwise_and(thres, hueMask(image, hue_min, hue_max))
    thres = cv2.bitwise_and(hueMask(image, hue_min, hue_max),hueMask(image, hue_min, hue_max))
    return thres

# przyjmuje wartosc w stopniach 
def getHue(hue):
    return (int)(hue /2)

# maska na dany kolor
def hueMask(img, lower, upper, neg=False):
    image   = cv2.GaussianBlur(img, (3,3), 0) 
    hsv     = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    l_col   = np.array([getHue(lower),85, 50])
    u_col   = np.array([getHue(upper),255,255])
    mask    = cv2.inRange(hsv, l_col, u_col)
    return cv2.bitwise_and(mask, mask)

def contourImage(filename, moments):
    image   = cv2.imread(filename)
    norm    = normalizeHist(image)
    thres   = [None] * TYPES
    for i in range(0, TYPES):
        thres[i]   = thresholding(norm, HUE_MIN, HUE_MAX)

    dil = cv2.dilate(thres[0], np.ones((3,3), 'uint8'), iterations=2)
    ero = cv2.erode(dil, np.ones((3,3), 'uint8'), iterations=2)

    cont, hier = cv2.findContours(ero, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    '''
    maxArea = 0;
    for c in cont:
        var = cv2.contourArea(c)
        if (var > maxArea):
            maxArea = var
            '''
    for i in range(0, len(cont)):
        #areas.append #
        bndX, bndY, bndW, bndH = cv2.boundingRect(cont[i])

        if (bndW * bndH * AREA < cv2.contourArea(cont[i])):
        #if (True):
            #approx  = cv2.approxPolyDP(cont[i], 0.01 * cv2.arcLength(cont[i], True), True)
            M       = cv2.moments(cont[i])
            huM     = cv2.HuMoments(M)
            for j in range(0,7):
                huM[j] = -1 * np.copysign(1.0, huM[j]) * np.log10(abs(huM[j]))
            if M['m00'] != 0.0:
                x = int(M['m10']/M['m00'])
                y = int(M['m01']/M['m00'])

            moments.append(huM)
            r, g, b = rand(50, 255), rand(50, 255), rand(50, 255)
            image = cv2.putText(image, str(i),
                    (x, y), cv2.FONT_HERSHEY_SIMPLEX, 
                       1, (r, g, b), 3, cv2.LINE_AA)
                       
            cv2.drawContours(image, cont, i, (r, g, b), 5)
            cv2.circle(image,(x, y), 2, (255,255,255), 5)
            #image = thres[0][bndY: bndY + bndH, bndX : bndX + bndW]
            #image = image[bndY: bndY + bndH, bndX : bndX + bndW]
            print(i)
            print(huM)
    return thres[0]

def main():
    if (len(sys.argv) < 2):
        print("please input data in correct format!")
        print("format: rsd.py <input image>")
    else:
        # momenty Hu z ktorymi bedziemy porownywac znalezione znaki
        contourImage('znaki/rondo.jpg', correct_moments) 
        print('-------------------------------')

        print(sys.argv[1:])
        fileList = sys.argv[1:]

        img = cv2.imread(fileList[0])
        image = contourImage(fileList[0], MOMENTS)
        if image is None:
            sys.exit("Could not read the image.")

        cv2.startWindowThread()
        cv2.namedWindow("floating")
        cv2.imshow("floating", image)
        while (cv2.waitKey(0) != ord("q")):
            print('')
        cv2.imwrite('output.png', image)

        '''
        fig = plt.figure(figsize=(8,8))

        for i in range(0, len(fileList)):
            image = contourImage(fileList[i], MOMENTS)
            #fig.add_subplot(3, 2, i + 1)
            fig.add_subplot(1,1,1)
            plt.axis('off')
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.savefig('output.png')
        '''
        print("Exiting...")

if (__name__ == "__main__"):
    main()
