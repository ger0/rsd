#!/bin/python
import numpy as np
import cv2
from random import randint as rand
from matplotlib import pyplot as plt

moments = []

# dziala?
def LaplacianOfGaussian(image):
    blur    = cv2.GaussianBlur(image, (3,3), 0)
    gray    = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    laplac  = cv2.Laplacian(gray, cv2.CV_8U,3,3,2)
    laplac  = cv2.convertScaleAbs(laplac)
    return laplac 
    
def thresholding(image):
    laplac = LaplacianOfGaussian(image)
    # 190 - 220 hue dla niebieskich znakow 
    thres = cv2.bitwise_and(laplac,laplac, mask=hueMask(image, 190, 220))
    thres = cv2.threshold(thres, 32, 255, cv2.THRESH_BINARY)[1]
    return thres

def getText(hum):
    strink = str('')
    for i in hum:
        strink += str(float(int(i) * 100) / 100) + '\n'
    return strink

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
    thres   = thresholding(image)
    #blur    = cv2.GaussianBlur(gray, (3,3), 0)

    ## statystyka, mozliwe ze nie warto
    #mn = np.mean(gray)
    #sd = np.std(gray)

    #thres = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    #thres = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    #thres = cv2.bitwise_and(thres, thres, mask=hueMask(image, 160, 260))

    #thres = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 10, 15)

    # nieco szersze krawedzie 

    #edged = cv2.Canny(ero, mn - 2 * sd, mn)

    hor = np.array([[0,1,0], [0, 1, 0], [0, 1, 0]], 'uint8')
    ver = np.array([[0,0,0], [1, 1, 1], [0, 0, 0]], 'uint8')

    #dil = cv2.dilate(thres, hor, iterations=2)
    #dil = cv2.dilate(dil, ver, iterations=2)
    dil = cv2.dilate(thres, np.ones((3,3), 'uint8'), iterations=2)
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
            '''
            M       = cv2.moments(cont[i])
            huM     = cv2.HuMoments(M)
            for j in range(0,7):
                huM[j] = -1 * np.copysign(1.0, huM[j]) * np.log10(abs(huM[j]))
            if M['m00'] != 0.0:
                x = int(M['m10']/M['m00'])
                y = int(M['m01']/M['m00'])

            moments.append(huM)
            image = cv2.putText(image, getText(huM),
                    (x, y), cv2.FONT_HERSHEY_SIMPLEX, 
                       1, (r, g, b), 3, cv2.LINE_AA)
            '''
            r, g, b = rand(50, 255), rand(50, 255), rand(50, 255)
            cv2.drawContours(image, cont, i, (r, g, b), 5)
            #cv2.circle(image,(x, y), 2, (0,0,0), 10)
    return image


fileList = ['sredni01.png']
fig = plt.figure(figsize=(8,8))

for i in range(0, len(fileList)):
    image = contourImage(fileList[i])
    #fig.add_subplot(3, 2, i + 1)
    fig.add_subplot(1,1,1)
    plt.axis('off')
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.savefig('output.png')
print("Exiting...")
