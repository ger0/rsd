import cv2
import numpy as np
import custom_types as ct

# poszukiwane ustawienia
BLUE_MIN    = 190
BLUE_MAX    = 250

RED_MIN     = 315
RED_MAX     = 374

HSV_SAT     = 73
HSV_VAL     = 70

# TODO: refactor
HSV_HUE = {
        ct.Colors.BLUE : (BLUE_MIN, BLUE_MAX),
        ct.Colors.RED  : (RED_MIN,  RED_MAX)
        }

def loadNorm(filename):
    img     = cv2.imread(filename)
    norm    = normalizeHist(img)
    calcAverages(norm)
    return img, norm

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
    print('Sat SD: ', np.std(S), 'Val SD:', np.std(V))

    if (mean_sat > HSV_SAT):
        HSV_SAT = int(mean_sat)

    if (mean_val - sd_val > HSV_VAL):
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
def threshold(img, color):
    (lower, upper) = HSV_HUE[color]
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

