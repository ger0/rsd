#!/bin/python
import sys
import numpy as np
import cv2
from matplotlib import pyplot as plt
import custom_types as ct
import preprocessing as pre
import classify as det

candidates      = []

AREA        = 0.6
isSetup     = True

def contourImage(filename, typ = None):
    image, norm = pre.loadNorm(filename, isSetup)
    thres, dil, ero  = {}, {}, {}

    # dla kazdego typu koloru
    for color in ct.Colors:
        # do debugowania
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
                    huM[k] = -1 * np.copysign(1.0, huM[k]) * np.log10(np.abs(huM[k]))
                if M['m00'] != 0.0:
                    x = int(M['m10']/M['m00'])
                    y = int(M['m01']/M['m00'])

                r, g, b = 0, 0, 0
                if (color == ct.Colors.BLUE):
                    r, g, b = 50, 50, 255
                elif (color == ct.Colors.RED):
                    r, g, b = 255, 50, 50
                '''
                elif (color == ct.Colors.YELLOW):
                    r, g, b = 255, 255, 50
                    '''

                cv2.drawContours(ero[color], cont, iter_cont, 255, -1)
                cropped = np.copy(norm[bndY: bndY + bndH, bndX : bndX + bndW])
                mask    = np.copy(ero[color][bndY: bndY + bndH, bndX : bndX + bndW])
                # wycinanie tla
                crop    = pre.crop(cropped, mask)

                # wstepna konfiguracja
                if (isSetup == True and typ != None):
                    det.loadContour(cont[iter_cont], typ) 
                    #det.loadHuMoment(huM, typ)
                    #det.loadHistogram(crop, typ) 
                    # wczytywanie histogramow
                else:
                   # print('distance crossw:', det.matchShapes(cont[iter_cont], ct.Type.crosswalk))
                   # print('distance rondo:', det.matchShapes(cont[iter_cont], ct.Type.roundabout))
                   # print('distance parking:', det.matchShapes(cont[iter_cont], ct.Type.parking))
                   # print('distance stop:', det.matchShapes(cont[iter_cont], ct.Type.stop))
                    #print('closest shape:', det.closestShape(cont[iter_cont]))
                    #print('closest histog:', det.closestHistogram(crop))
                    #det.closestHistogram(crop)
                # draw
                    shapeMatch, shapeDist = det.closestShape(cont[iter_cont], color)
                    if (shapeDist < 0.01):
                        print(shapeMatch, shapeDist)
                        #debug
                        plt.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                        plt.show()
                        #
                        image = cv2.putText(image, str(color),
                                (x, y), cv2.FONT_HERSHEY_SIMPLEX, 
                                   1, (b, g, r), 3, cv2.LINE_AA)
                        cv2.drawContours(image, cont, iter_cont, (b, g, r), 5)
                        cv2.circle(image,(x, y), 2, (255,255,255), 5)
    return ero[ct.Colors.BLUE]
    
def main():
    if (len(sys.argv) < 2):
        print("please input data in correct format!")
        print("format: rsd.py <input image>")
    else:
        print('-------------------------------')

        fileList = sys.argv[1:]

        # wczytywanie konturow dla zdjec referencyjnych
        contourImage('referencja/rondo.jpg', ct.Type.circular)
        contourImage('referencja/parking.jpg', ct.Type.square)
        isSetup = False

        ''' # nie dziala na PC????
        cv2.imshow("floating", image)
        while (cv2.waitKey(0) != ord("q")):
            print('')
        cv2.imwrite('output.png', image)

        '''
        fig = plt.figure(figsize=(8,8))

        for i in range(0, len(fileList)):
            print(fileList[i])
            image = contourImage(fileList[i])
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
