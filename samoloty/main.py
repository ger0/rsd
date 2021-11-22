#!/bin/python
import numpy as np
import cv2
from random import randint as rand
from matplotlib import pyplot as plt

def contourImage(filename):
    image = cv2.imread(filename)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    # statystyka, mozliwe ze nie warto
    mn = np.mean(gray)
    sd = np.std(gray)

    # nie pomagaja
    ro, thres = cv2.threshold(blur, mn - sd, 255, cv2.THRESH_BINARY_INV)
    #thres = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 15)

    edged = cv2.Canny(thres, mn - 2 * sd, mn)

    # nieco szersze krawedzie 
    dil = cv2.dilate(edged, np.ones((3,3), 'uint8'), iterations=2)
    ero = cv2.erode(dil, np.ones((3,3), 'uint8'), iterations=1)

    cont, hier = cv2.findContours(ero, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for i in range(0, len(cont)):
        approx  = cv2.approxPolyDP(cont[i], 0.01 * cv2.arcLength(cont[i], True), True)

        M       = cv2.moments(cont[i])
        huM    = cv2.HuMoments(M)
        for j in range(0,7):
	        huM[j] = -1 * np.copysign(1.0, huM[j]) * np.log10(abs(huM[j]))
        if M['m00'] != 0.0:
            x = int(M['m10']/M['m00'])
            y = int(M['m01']/M['m00'])

        #print(cv2.contourArea(cont[i]))
        print(huM)
        cv2.drawContours(image, cont, i, (rand(50,255), rand(50,255), rand(50,255)), 5)
        cv2.circle(image,(x, y), 2, (255,255,255), 10)
    return image

fileList = ['../trojkat.png']
'''
fileList = ['samolot06.jpg', 'samolot11.jpg', 'samolot07.jpg',
            'samolot14.jpg', 'samolot04.jpg', 'samolot16.jpg']
'''
fig = plt.figure(figsize=(8,8))

for i in range(0, len(fileList)):
    image = contourImage('photo/' + fileList[i])
    #fig.add_subplot(3, 2, i + 1)
    fig.add_subplot(1,1,1)
    plt.axis('off')
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    print('___________________________________')
plt.savefig('output.png')
print("Exiting...")
