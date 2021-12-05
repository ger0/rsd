import custom_types as ct
from matplotlib import pyplot as plt
import numpy as np
import cv2

hu_moments = {}

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
    hu_moment[typ] = parsed
    print(hu_moment[typ])

def calcHistogram(img, color):
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()

      
