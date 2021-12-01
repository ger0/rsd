#!/bin/python
import sys
import numpy as np
import cv2
from random import randint as rand
from matplotlib import pyplot as plt

# liczba znakow 
TYPES   = 1

# poszukiwane ustawienia
HUE_MIN     = 190
HUE_MAX     = 220
AREA        = 0.5
MOMENTS     = []
THRESHOLD   = 32

# dziala?
def LaplacianOfGaussian(image):
    blur    = cv2.GaussianBlur(image, (3,3), 0)
    gray    = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    laplac  = cv2.Laplacian(gray, cv2.CV_8U,3,3,2)
    laplac  = cv2.convertScaleAbs(laplac)
    return laplac 
    
def thresholding(image, hue_min, hue_max):
    laplac = LaplacianOfGaussian(image)
    thres = cv2.bitwise_and(laplac,laplac, mask=hueMask(image, hue_min, hue_max))
    thres = cv2.threshold(thres, THRESHOLD, 255, cv2.THRESH_BINARY)[1]
    return thres

def getText(hum):
    string = str('')
    for i in hum:
        string += str(float(int(i) * 100) / 100) + '\n'
    return string

# przyjmuje wartosc w stopniach 
def getHue(hue):
    return (int)(hue /2)

# maska na dany kolor
def hueMask(img, lower, upper):
    image   = cv2.GaussianBlur(img, (3,3), 0) 
    hsv     = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    l_col   = np.array([getHue(lower),50, 50])
    u_col   = np.array([getHue(upper),255,255])
    mask    = cv2.inRange(hsv, l_col, u_col)
    return cv2.bitwise_and(mask, mask)

def contourImage(filename):
    image   = cv2.imread(filename)
    thres   = [None] * TYPES
    for i in range(0, TYPES):
        thres[i]   = thresholding(image, HUE_MIN, HUE_MAX)

    #thres = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    #thres = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    #thres = cv2.bitwise_and(thres, thres, mask=hueMask(image, 160, 260))
    #thres = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 10, 15)

    # nieco szersze krawedzie 
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

        if (bndW * bndH * 0.5 < cv2.contourArea(cont[i])):
        #if (True):
            #approx  = cv2.approxPolyDP(cont[i], 0.01 * cv2.arcLength(cont[i], True), True)
            M       = cv2.moments(cont[i])
            huM     = cv2.HuMoments(M)
            for j in range(0,7):
                huM[j] = -1 * np.copysign(1.0, huM[j]) * np.log10(abs(huM[j]))
            if M['m00'] != 0.0:
                x = int(M['m10']/M['m00'])
                y = int(M['m01']/M['m00'])

            MOMENTS.append(huM)
            r, g, b = rand(50, 255), rand(50, 255), rand(50, 255)
            '''image = cv2.putText(image, getText(huM),
                    (x, y), cv2.FONT_HERSHEY_SIMPLEX, 
                       1, (r, g, b), 3, cv2.LINE_AA)
                       '''
            cv2.drawContours(image, cont, i, (r, g, b), 5)
            #cv2.circle(image,(x, y), 2, (0,0,0), 10)
            #image = thres[0][bndY: bndY + bndH, bndX : bndX + bndW]
            #image = image[bndY: bndY + bndH, bndX : bndX + bndW]
            print(huM)
    return image

def main():
    if (len(sys.argv) < 2):
        print("please input data in correct format!")
        print("format: rsd.py <input image>")
    else:
        print(sys.argv[1:])
        fileList = sys.argv[1:]
        fig = plt.figure(figsize=(8,8))

        for i in range(0, len(fileList)):
            image = contourImage(fileList[i])
            #fig.add_subplot(3, 2, i + 1)
            fig.add_subplot(1,1,1)
            plt.axis('off')
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.savefig('output.png')
        print("Exiting...")

if (__name__ == "__main__"):
    main()
