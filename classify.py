import custom_types as ct
from matplotlib import pyplot as plt
import numpy as np
import cv2

hu_moments  = {}
contours    = {}

#def loadValues:
#def compare

def saveVal(filename, val):
    file    = open(filename, 'w')
    for i in val:
        [a] = i
        file.write(str(a) + ' ')
    file.close()

def loadHuMoment(filename, typ):
    file    = open(filename, 'r')
    val = file.read()
    file.close()
    splt = val.split(' ')
    parsed = np.array(splt[0:-1], dtype=float)
    hu_moments[typ] = parsed
    print(hu_moments[typ])

def loadContour(contour, typ):
   contours[typ] = contour 

def calcHistogram(img, color):
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()

# oblicza dystans pomiedzy dwoma wektorami momentow
# rip nie dziala jak powinno
'''
def matchShapes(h1, typ):
    h2 = hu_moments[typ]
    if (len(h1) != 7 or len(h2) != 7):
        print("ERROR while matching shapes")
        print('H1:', len(h1), 'H2:', len(h2))
        return -1
    else:
        dist = 0
        for i in range(0, 7):
            dist += np.abs(h1[i] - h2[i])
        return dist
'''
def matchShapes(cont, typ):
    return cv2.matchShapes(cont, contours[typ], cv2.CONTOURS_MATCH_I2, 0)

def closestShape(cont):
    min_dist = 9999999
    min_type = None
    for key, value in contours.items():
        dist = matchShapes(cont, key)
        if (dist < min_dist):
            min_dist = dist
            min_type = key

    return min_type

