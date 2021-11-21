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
    ro, thres = cv2.threshold(blur, 130, 255, cv2.THRESH_BINARY_INV)
    #thres = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 15)

    edged = cv2.Canny(thres, mn - 2 * sd, mn)

    # nieco szersze krawedzie 
    dil = cv2.dilate(edged, np.ones((3,3), 'uint8'), iterations=2)
    ero = cv2.erode(dil, np.ones((3,3), 'uint8'), iterations=1)

    cont, hier = cv2.findContours(ero, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for i in range(0, len(cont)):
        x, y = 0, 0
        cv2.drawContours(image, cont, i, (rand(50,255), rand(50,255), rand(50,255)), 5)
        for j in range(0, int(len(cont[i]))):
            x = x + cont[i][j][0][0]
            y = y + cont[i][j][0][1]
        x = int(x / (len(cont[i])))
        y = int(y / (len(cont[i])))
        cv2.circle(image,(x, y), 2, (255,255,255), 10)
    return image


fileList = ['sredni01.png']
for i in range(0, len(fileList)):
    image = contourImage(fileList[i])
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.savefig('output.png')
print("Exiting...")
