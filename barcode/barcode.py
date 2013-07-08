'''
Created on May 18, 2013

@author: Luca
'''

import os 
import cv2
import numpy as np
import time
from BarcodeTools import BarcodeTools

import zbar

if __name__ == '__main__':
    scanner = zbar.ImageScanner()
    scanner.parse_config('enable')
    bt = BarcodeTools(True)
    
    for dirname, dirnames, filenames in os.walk("d:\\barcode"):
        if dirname != 'd:\\barcode\\processed' and dirname != 'd:\\barcode\\processed_bck':            
            for filename in filenames:
            #dirname = 'd:\\barcode\jpg'
            #filename = '00052322.jpg'
                start = time.clock()
                                         
                img = cv2.imread(os.path.join(dirname, filename), cv2.CV_LOAD_IMAGE_COLOR)
                contours, bcode_imgs = bt.detect(img)
                  
                for bcode_img, contour in zip(bcode_imgs, contours):
                    box = np.int0(cv2.cv.BoxPoints(contour))
                    cv2.drawContours(img, [box], 0, (0, 255, 0), thickness = 2)
                      
                    cv2.putText(img, bt.decode(bcode_img), (contour[0][0], contour[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), thickness=2) 
                 
                end = time.clock()
                 
                outname = os.path.join('d:\\barcode', 'processed', filename)
                cv2.imwrite(outname, img)
                print "{0}: time {1}".format(outname, end - start)                