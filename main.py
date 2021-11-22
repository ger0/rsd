#!/bin/python
import numpy as np
import cv2
from random import randint as rand
from matplotlib import pyplot as plt

moments = []

def getText(hum):
    strink = str('')
    for i in hum:
        strink += str(float(int(i) * 100) / 100) + '\n'
    return strink

def getHue(hue):
    return (int)(hue * (255 / 360))

# maska na dany kolor
def hueMask(img, lower, upper):
    image   = cv2.GaussianBlur(img, (3,3), 0) 
    hsv     = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    l_col = np.array([getHue(lower),128,0])
    u_col = np.array([getHue(upper),255,255])

    l_white = np.array([0,0,128])
    u_white = np.array([255,255,255])

    mask_col    = cv2.inRange(hsv, l_col, u_col)
    #return mask_col

    l_white = np.array([0,0,128], dtype=np.uint8)
    u_white = np.array([255,255,255], dtype=np.uint8)

    l_black = np.array([0,0,0], dtype=np.uint8)
    u_black = np.array([170,150,50], dtype=np.uint8)

    mask_white = cv2.inRange(hsv, l_white, u_white)
    mask_black = cv2.inRange(hsv, l_black, u_black)

    mask = cv2.bitwise_or(mask_col, mask_white)
    # brak czarnej maski byc moze kluczowy

    return mask

def contourImage(filename):
    image   = cv2.imread(filename)
    gray    = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur    = cv2.GaussianBlur(gray, (3,3), 0)

    # statystyka, mozliwe ze nie warto
    mn = np.mean(gray)
    sd = np.std(gray)

    #thres = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY +cv2.THRESH_OTSU)[1]
    thres = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    thres = cv2.bitwise_and(thres,thres, mask=hueMask(image, 71, 165))

    #thres = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 10, 15)

    # nieco szersze krawedzie 
    dil = cv2.dilate(thres, np.ones((3,3), 'uint8'), iterations=2)
    ero = cv2.erode(dil, np.ones((3,3), 'uint8'), iterations=1)

    #edged = cv2.Canny(ero, mn - 2 * sd, mn)
    edged = cv2.Canny(ero, 0, 255)

    cont, hier = cv2.findContours(ero, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #areas = []
    for i in range(0, len(cont)):
        #approx  = cv2.approxPolyDP(cont[i], 0.01 * cv2.arcLength(cont[i], True), True)
        #areas.append 
        '''
        M       = cv2.moments(cont[i])
        huM     = cv2.HuMoments(M)
        for j in range(0,7):
	        huM[j] = -1 * np.copysign(1.0, huM[j]) * np.log10(abs(huM[j]))
        if M['m00'] != 0.0:
            x = int(M['m10']/M['m00'])
            y = int(M['m01']/M['m00'])

        moments.append(huM)
        '''
        r, g, b = rand(50, 255), rand(50, 255), rand(50, 255)
        cv2.drawContours(image, cont, i, (r, g, b), 5)
        #cv2.circle(image,(x, y), 2, (0,0,0), 10)
        '''
        image = cv2.putText(image, getText(huM),
                (x, y), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (r, g, b), 3, cv2.LINE_AA)
                   '''
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
