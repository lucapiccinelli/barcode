'''
Created on May 25, 2013

@author: Luca
'''

import cv2
import sys
import numpy as np    
from BarcodeTools import BarcodeTools, BarcodeInfo

CV_CAP_PROP_FRAME_WIDTH = 3
CV_CAP_PROP_FRAME_HEIGHT = 4
FRAME_COUNT = 5

if __name__ == '__main__':
    
    cap = cv2.VideoCapture(0) 
    
    if not cap.isOpened():
        print "dev not opened"
        sys.exit(-1)
        
    barCodeTools = BarcodeTools(True) 
    
    cv2.namedWindow("bla")
    cv2.namedWindow("bla2")
    cv2.namedWindow("barcode")
    
    w = 1024
    h = 768
    i = 0
    prev_frame = None
    prev_imgs = []
    
    bcodes_info = {}
    while True:
        x = y = 0
        extracted_bcodes = np.zeros((h, w), np.uint8)
        s_extracted_bcodes = np.zeros((h, w), np.uint8)    
        
        if i == FRAME_COUNT:
            cap.set(CV_CAP_PROP_FRAME_WIDTH,  1920)
            cap.set(CV_CAP_PROP_FRAME_HEIGHT, 1080)
            
        i += 1
        
        cap.grab()
        success, frame = cap.retrieve()
        
        if i > FRAME_COUNT:
            contours, bcode_imgs, shocked_bcode_imgs, center = barCodeTools.detect(frame)
              
            bcodes_num = len(bcode_imgs)
            decoded_num = 0
            if bcodes_num > 0:
                matches = barCodeTools.match(bcode_imgs, prev_imgs)
                for bcode_img, s_bcode_img, contour, match in zip(bcode_imgs, shocked_bcode_imgs, contours, matches):
                    if match != None:
                        bcode_i = next(bcode_i for bcode_i in bcodes_info.values() if id(bcode_i.bcode_img) == id(match))
                        bcode_i.bcode_img = bcode_img
                        bcode_i.contour   = contour
                        
                        if bcode_i.decode_str == '':
                            decode1 = barCodeTools.decode(s_bcode_img)
                            decode2 = barCodeTools.decode(bcode_img)
            
                            bcode_i.decode_str = decode1 if decode1 != '' else decode2
                    else:
                        decode1 = barCodeTools.decode(s_bcode_img)
                        decode2 = barCodeTools.decode(bcode_img)
                        
                        bcode_i = BarcodeInfo(bcode_img, decode1 if decode1 != '' else decode2, contour)
                        bcodes_info[bcode_i.ID] = bcode_i
                        
            for bcode_i in bcodes_info.values():
                box = np.int0(cv2.cv.BoxPoints(bcode_i.contour))
                cv2.drawContours(frame, [box], 0, bcode_i.color, thickness = 2)
                
                cv2.putText(frame, bcode_i.decode_str, (contour[0][0], contour[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), thickness=2)          
                        
            prev_imgs = bcode_imgs
            
        prev_frame = frame
        
        cv2.imshow("barcode", frame)
        if cv2.waitKey(30) == 27: break
        
    cv2.destroyAllWindows()
    
    
    

#     cv2.namedWindow("extracted_bcodes")
#     cv2.cv.CreateTrackbar("sigma", "extracted_bcodes", barCodeTools.shock_sigma, 15, barCodeTools.set_shock_sigma)
#     cv2.cv.CreateTrackbar("str_sigma", "extracted_bcodes", barCodeTools.shock_str_sigma, 15, barCodeTools.set_shock_str_sigma)
#     cv2.cv.CreateTrackbar("blend", "extracted_bcodes", barCodeTools.shock_blend * 10, 10, lambda x: barCodeTools.set_shock_blend(x / 10.0))
#     cv2.cv.CreateTrackbar("iter", "extracted_bcodes", barCodeTools.shock_iter, 8, barCodeTools.set_shock_iter)
#     cv2.cv.CreateTrackbar("thresh", "extracted_bcodes", barCodeTools.shock_thresh, 255, barCodeTools.set_shock_thresh)
    
    
#         cv2.imshow("extracted_bcodes", extracted_bcodes)
#         cv2.imshow("s_extracted_bcodes", s_extracted_bcodes)
    
    
#     for bcode_img, s_bcode_img, contour in zip(bcode_imgs, shocked_bcode_imgs, contours):
#             box = np.int0(cv2.cv.BoxPoints(contour))
#             cv2.drawContours(frame, [box], 0, (0, 255, 0), thickness = 2)
#               
#             if x + bcode_img.shape[1] > w:
#                 x = 0
#                 y += 100
#             extracted_bcodes[y:y + bcode_img.shape[0], x:x + bcode_img.shape[1]] = bcode_img
#             s_extracted_bcodes[y:y + bcode_img.shape[0], x:x + bcode_img.shape[1]] = s_bcode_img
#             x += bcode_img.shape[1] + 10
#             
#             decoded1 = barCodeTools.decode(s_bcode_img)
#             decoded2 = barCodeTools.decode(bcode_img)
#               
#             cv2.putText(frame, decoded1, (contour[0][0], contour[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), thickness=2)
#             cv2.putText(frame, decoded2, (contour[0][0], contour[0][1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), thickness=2)