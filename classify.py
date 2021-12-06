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

def loadHistogram(histogram, typ):
    histograms[typ] = histogram

def calcHistogram(img, color):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    (h, s, v) = cv2.split(hsv)
    hist = cv2.calcHist(h, [0], None, [10], (0, 256))

    histRange = (0, 256)
    histSize = 256
    hist_w = 512
    hist_h = 400
    bin_w = int(round(hist_w/histSize))

    b_hist = cv2.calcHist(img, [0], None, [histSize], histRange, accumulate=False)
    g_hist = cv2.calcHist(img, [1], None, [histSize], histRange, accumulate=False)
    r_hist = cv2.calcHist(img, [2], None, [histSize], histRange, accumulate=False)

    cv2.normalize(b_hist, b_hist, alpha=0, beta=hist_h, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(g_hist, g_hist, alpha=0, beta=hist_h, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(r_hist, r_hist, alpha=0, beta=hist_h, norm_type=cv2.NORM_MINMAX)

    histImage = np.zeros((hist_h, hist_w, 3), dtype=np.uint8)

    for i in range(1, histSize):
        cv2.line(histImage, ( bin_w*(i-1), hist_h - int(b_hist[i-1]) ),
                (bin_w*(i), hist_h - int(b_hist[i]) ),
                (255, 0, 0), thickness=2)
        '''
        cv2.line(histImage, ( bin_w*(i-1), hist_h - int(g_hist[i-1]) ),
                (bin_w*(i), hist_h - int(g_hist[i]) ),
                (0, 255, 0), thickness=2)
                '''
        cv2.line(histImage, ( bin_w*(i-1), hist_h - int(r_hist[i-1]) ),
                (bin_w*(i), hist_h - int(r_hist[i]) ),
                (0, 0, 255), thickness=2)

    # rysowanie znaku
    plt.imshow(cv2.cvtColor(histImage, cv2.COLOR_BGR2RGB))
    plt.show()
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

