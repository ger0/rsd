#!/bin/python
from enum import Enum
import sys
import numpy as np
from random import randint as rand
import cv2
from matplotlib import pyplot as plt

# liczba znakow 
TYPES   = 2

class Types(Enum):
    INFO    =   0
    STOP    =   1

correct_moments = []

# poszukiwane ustawienia
BLUE_MIN    = 190
BLUE_MAX    = 250

RED_MIN     = 325
RED_MAX     = 374

HSV_SAT     = 73
HSV_VAL     = 70
AREA        = 0.5
MOMENTS     = []
THRESHOLD   = 42

def calcAverages(image):
    global HSV_SAT, HSV_VAL
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    (H, S, V) = cv2.split(hsv)

    mean_sat = np.mean(S)
    sd_sat  = np.std(S)

    mean_val = np.mean(V)
    sd_val  = np.std(V)
    print(
          'Hue:',
          np.mean(H),
          'Sat:',
          mean_sat,
          'Val:',
          mean_val
          )
    print('Hue SD: ', np.std(S), 'Sat SD:', np.std(V))

    if (mean_sat > HSV_SAT):
        HSV_SAT = int(mean_sat)

    if (mean_val - sd_val > HSV_VAL):
        print('change')
        HSV_VAL = int(mean_val - sd_val)

def normalizeHist(image):
    (B, G, R) = cv2.split(image)
    clahe = cv2.createCLAHE(clipLimit=0.4, tileGridSize=(8,8))
    #B = cv2.equalizeHist(B)
    #G = cv2.equalizeHist(G)
    #R = cv2.equalizeHist(R)
    B = clahe.apply(B)
    G = clahe.apply(G)
    R = clahe.apply(R)
    #return normalizedimage
    return cv2.merge((B, G, R))
'''
# dziala?
def LaplacianOfGaussian(image):
    blur    = cv2.GaussianBlur(image, (3,3), 0)
    gray    = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    laplac  = cv2.Laplacian(gray, cv2.CV_8U,3,3,2)
    laplac  = cv2.convertScaleAbs(laplac)
    return laplac 
    '''

# przyjmuje wartosc w stopniach (0 - 720)
def getHue(hue):
    if (hue > 360):
        return (int)((hue - 360) / 2), True 
    else:
        return (int)(hue / 2), False 

# maska na dany kolor
def thresholding(img, lower, upper):
    image   = cv2.GaussianBlur(img, (9,9), 0) 
    hsv     = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    l, _ = getHue(lower)
    u, isOverlap = getHue(upper)

    # TODO: oczyscic funkcje
    if (isOverlap):
        l_col = np.array([l, HSV_SAT, HSV_VAL])
        u_col = np.array([180, 255, 255])

        l1_col  = np.array([0, HSV_SAT, HSV_VAL])
        u1_col  = np.array([u, 255, 255]) 

        mask    = cv2.inRange(hsv, l_col, u_col)
        mask1   = cv2.inRange(hsv, l1_col, u1_col)
        return cv2.bitwise_or(mask, mask1)
    else:
        l_col = np.array([l, HSV_SAT, HSV_VAL])
        u_col = np.array([u, 255, 255])

        mask    = cv2.inRange(hsv, l_col, u_col)
        return cv2.bitwise_and(mask, mask)

def contourImage(filename, moments):
    image   = cv2.imread(filename)
    norm    = normalizeHist(image)
    calcAverages(norm)
    thres   = [None] * TYPES
    dil     = [None] * TYPES
    ero     = [None] * TYPES

    # dla kazdego typu koloru
    for i in range(0, TYPES):
        if (i == 0):
            thres[i]    = thresholding(norm, BLUE_MIN, BLUE_MAX)
        elif (i == 1):
            # overlap : 420 = 360 + 60
            thres[i]    = thresholding(norm, RED_MIN, RED_MAX)

        dil[i]  = cv2.dilate(thres[i], np.ones((3,3), 'uint8'), iterations=2)
        ero[i]  = cv2.erode(dil[i], np.ones((3,3), 'uint8'), iterations=2)

        cont, hier = cv2.findContours(ero[i], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for iter_cont in range(0, len(cont)):
            #areas.append #
            bndX, bndY, bndW, bndH = cv2.boundingRect(cont[iter_cont])

            if (bndW * bndH * AREA < cv2.contourArea(cont[iter_cont])):
            #if (True):
                #approx  = cv2.approxPolyDP(cont[i], 0.01 * cv2.arcLength(cont[i], True), True)
                M       = cv2.moments(cont[iter_cont])
                huM     = cv2.HuMoments(M)
                for k in range(0,7):
                    huM[k] = -1 * np.copysign(1.0, huM[k]) * np.log10(abs(huM[k]))
                if M['m00'] != 0.0:
                    x = int(M['m10']/M['m00'])
                    y = int(M['m01']/M['m00'])

                moments.append(huM)
                r, g, b = 0, 0, 0
                if (i == 0):
                    r, g, b = 50, 50, 255
                elif (i == 1):
                    r, g, b = 255, 50, 50

                # draw
                image = cv2.putText(image, str(i),
                        (x, y), cv2.FONT_HERSHEY_SIMPLEX, 
                           1, (b, g, r), 3, cv2.LINE_AA)
                cv2.drawContours(image, cont, iter_cont, (b, g, r), 5)
                cv2.circle(image,(x, y), 2, (255,255,255), 5)
                #image = thres[0][bndY: bndY + bndH, bndX : bndX + bndW]
                #image = image[bndY: bndY + bndH, bndX : bndX + bndW]

                #print(i)
                #print(huM)
    return image
    
def main():
    if (len(sys.argv) < 2):
        print("please input data in correct format!")
        print("format: rsd.py <input image>")
    else:
        # momenty Hu z ktorymi bedziemy porownywac znalezione znaki
        #contourImage('znaki/rondo.jpg', correct_moments) 
        print('-------------------------------')

        fileList = sys.argv[1:]

        ''' # nie dziala na PC????
        cv2.imshow("floating", image)
        while (cv2.waitKey(0) != ord("q")):
            print('')
        cv2.imwrite('output.png', image)

        '''
        fig = plt.figure(figsize=(8,8))

        for i in range(0, len(fileList)):
            print(fileList[i])
            image = contourImage(fileList[i], MOMENTS)
            if image is None:
                sys.exit("Could not read the image.")
            #fig.add_subplot(3, 2, i + 1)
            fig.add_subplot(1,1,1)
            plt.axis('off')
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.savefig('output.png')
        print("Exiting...")

if (__name__ == "__main__"):
    main()
