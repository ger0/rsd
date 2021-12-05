#!/bin/python
import sys
import numpy as np
import cv2
from matplotlib import pyplot as plt
import custom_types as ct
import preprocessing as pre

print(repr(ct.Colors.RED))
correct_moments = []

candidates      = []

AREA        = 0.5
MOMENTS     = []
THRESHOLD   = 42

def contourImage(filename, moments):
    image, norm = pre.loadNorm(filename)
    thres   = {}
    dil     = {}
    ero     = {}

    # dla kazdego typu koloru
    for color in ct.Colors:
        thres[color] = pre.threshold(norm, color)
        dil[color]  = cv2.dilate(thres[color], np.ones((3,3), 'uint8'), iterations=2)
        ero[color]  = cv2.erode(dil[color], np.ones((3,3), 'uint8'), iterations=2)

        cont, hier = cv2.findContours(ero[color], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        for iter_cont in range(0, len(cont)):
            # bounding box
            bndX, bndY, bndW, bndH = cv2.boundingRect(cont[iter_cont])

            if (bndW * bndH * AREA < cv2.contourArea(cont[iter_cont])):
            #if (True):
                M       = cv2.moments(cont[iter_cont])
                huM     = cv2.HuMoments(M)
                for k in range(0,7):
                    huM[k] = -1 * np.copysign(1.0, huM[k]) * np.log10(abs(huM[k]))
                if M['m00'] != 0.0:
                    x = int(M['m10']/M['m00'])
                    y = int(M['m01']/M['m00'])

                moments.append(huM)
                r, g, b = 0, 0, 0
                if (color == ct.Colors.BLUE):
                    r, g, b = 50, 50, 255
                elif (color == ct.Colors.RED):
                    r, g, b = 255, 50, 50

                candidates.append(np.copy(image[bndY: bndY + bndH, bndX : bndX + bndW]))

                # draw
                image = cv2.putText(image, str(color),
                        (x, y), cv2.FONT_HERSHEY_SIMPLEX, 
                           1, (b, g, r), 3, cv2.LINE_AA)
                cv2.drawContours(image, cont, iter_cont, (b, g, r), 5)
                cv2.circle(image,(x, y), 2, (255,255,255), 5)
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
