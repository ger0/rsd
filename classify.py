import custom_types as ct
from matplotlib import pyplot as plt
import numpy as np
import cv2

hu_moments  = {}

contours    = {}
histograms  = {}

#def loadValues:
#def compare

def saveVal(filename, val):
    file    = open(filename, 'w')
    for i in val:
        [a] = i
        file.write(str(a) + ' ')
    file.close()

def loadHuMoment(hu, typ):
    file    = open(filename, 'r')
    val = file.read()
    file.close()
    splt = val.split(' ')
    parsed = np.array(splt[0:-1], dtype=float)
    hu_moments[typ] = parsed
    print(hu_moments[typ])
    hu_moments[typ] = hu

def loadContour(contour, typ):
   contours[typ] = contour 

def loadHistogram(img, typ):
    histogram = calcHistogram(img)
    histograms[typ] = histogram

def calcHistogram(img):
    #hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([img], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
    #hist = cv2.calcHist([hsv], [1, 2], None, [256, 256], [0, 256, 0, 256])
    return hist

def matchHistogram(hist, key):
    return cv2.compareHist(hist, histograms[key], 4)

def closestHistogram(img):
    min_dist = 9999999
    min_type = None
    for key, value in histograms.items():
        hist = calcHistogram(img)
        dist = matchHistogram(hist, key)
        if (dist < min_dist):
            min_dist = dist
            min_type = key

    print ('Closest histo:',  min_type, min_dist)
    #plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    #plt.show()
    return min_type

'''
# oblicza dystans pomiedzy dwoma wektorami momentow
# nie dziala jak powinno
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

# oblicza podobienstwo dwoch konturow
def matchShapes(cont, typ):
    return cv2.matchShapes(cont, contours[typ], cv2.CONTOURS_MATCH_I2, 0)

def closestShape(cont, color):
    min_dist = 9999999
    min_type = None
    for key, value in contours.items():
        dist = matchShapes(cont, key)
        if (dist < min_dist):
            if (not(color == ct.Colors.RED and key == ct.Type.square)):
                min_dist = dist
                min_type = key

    return min_type, min_dist
