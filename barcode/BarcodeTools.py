'''
Created on Jun 1, 2013

@author: Luca
'''

import cv2
import zbar
import numpy as np
import math
from math import cos, sin
from random import randint
from operator import itemgetter
import sys


def shock_filter(img, sigma = 11, str_sigma = 11, blend= 0.5, iter_n = 4, thresh = 70):
    h, w = img.shape[:2]
    blurred_gray = cv2.GaussianBlur(img, (5, 5), 1.5)
    img = cv2.addWeighted(img, 1.5, blurred_gray, -0.5, 0)
    t, img = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)
    
    for i in range(iter_n):
        eigen = cv2.cornerEigenValsAndVecs(img, str_sigma, 3)
        eigen = eigen.reshape(h, w, 3, 2)
        x, y = eigen[:, :, 1, 0], eigen[:, :, 1, 1] 
        
        gxx = cv2.Sobel(img, cv2.CV_32F, 2, 0, ksize = sigma)
        gxy = cv2.Sobel(img, cv2.CV_32F, 1, 1, ksize = sigma)
        gyy = cv2.Sobel(img, cv2.CV_32F, 0, 2, ksize = sigma)
        gvv = x * x * gxx + 2 * x * y * gxy + y * y * gyy
        m = gvv < 0
        
        ero = cv2.erode(img, None)
        dil = cv2.dilate(img, None)
        img1 = ero
        img1[m] = dil[m]
        img = np.uint8(img * (1.0 - blend) + img1 * blend)
    
    #return cv2.morphologyEx(img, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1)), iterations=1)
    return img
    

class BarcodeTools(object): 

    def __init__(self, sharpen = False):
        '''
        Constructor
        '''
        self.sharpen = sharpen
        self.scanner = zbar.ImageScanner()
        self.scanner.parse_config('enable')
        
        self.shock_sigma = 5
        self.shock_str_sigma = 7
        self.shock_blend = 0.5
        self.shock_iter = 1
        self.shock_thresh = 100
        
        self.stored_imgs = {}
            
    def __calc_connected_img(self, gray_img, structure_size_w, structure_size_h):
        step1_img = cv2.morphologyEx(gray_img, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (6, 1)), iterations=2)
        #cv2.imshow("ble", step1_img)
        step1_img = cv2.add(step1_img, cv2.Sobel(step1_img , -1, 1, 0))
        step1_img = cv2.GaussianBlur(step1_img, (3, 3), 1)
        #cv2.imshow("bla", step1_img)
        t, tresh_img = cv2.threshold(step1_img, 90, 255, cv2.THRESH_BINARY)
        #cv2.imshow("bla2", tresh_img)
        step2_img = cv2.morphologyEx(tresh_img, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (structure_size_w, 1)), iterations=1)
        #cv2.imshow("bla3", step2_img)
        step3_img = cv2.morphologyEx(step2_img, cv2.MORPH_OPEN,  cv2.getStructuringElement(cv2.MORPH_RECT, (structure_size_w, structure_size_h)), iterations=2)
        #cv2.imshow("bla4", step3_img)
        connected_img = cv2.dilate(step3_img, cv2.getStructuringElement(cv2.MORPH_RECT, (structure_size_w, structure_size_h)), iterations=1)
        del t
        return connected_img          
        
    def detect(self, img):
        out_bounding_rect = []
        out_bcode_imgs = []
        out_shocked_bcode_imgs = []
        
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if self.sharpen:
            blurred_gray = cv2.GaussianBlur(gray_img, (5, 5), 1.5)
            gray_img = cv2.addWeighted(gray_img, 1.5, blurred_gray, -0.5, 0)
            
        w = 800
        scale = float(w) / gray_img.shape[1] 
        h = int(gray_img.shape[0] * scale)  
        gray_img1 = cv2.resize(gray_img, (w, h), gray_img, cv2.INTER_CUBIC)
        scale = 1 / scale
            
        structure_size_w = 5
        structure_size_h = 5#int(structure_size_w * (w / h))
        connected_img = self.__calc_connected_img(gray_img1, structure_size_w, structure_size_h)
        
        contours, hierarchy = cv2.findContours(connected_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE, offset=(structure_size_w,structure_size_h))
        
        del hierarchy
        if contours != None:
            for contour in contours:
                if cv2.contourArea(contour) <= 1500:
                    continue
                
                rect = cv2.minAreaRect(contour)
                bRect = cv2.boundingRect(contour)
                bRect = (int(bRect[0] * scale), int(bRect[1] * scale), int(bRect[2] * scale), int(bRect[3] * scale))
                
                rect = ((int(rect[0][0] * scale), int(rect[0][1] * scale)), (int(rect[1][0] * scale), int(rect[1][1] * scale)), rect[2])
                roi_rect = rect
                
                angle = rect[2]
                if angle < -45:
                    angle += 90
                    roi_rect = (rect[0][0], rect[0][1]), (rect[1][1], rect[1][0]), rect[2]
                    
                roi = (bRect[2] / 2 - roi_rect[1][0] / 2, bRect[3] / 2 - roi_rect[1][1] / 2, roi_rect[1][0], roi_rect[1][1])
                roi = [(lambda x: x if x >= 0 else 0)(x) for x in roi]
                                    
                rotation_M = cv2.getRotationMatrix2D((bRect[2] / 2, bRect[3] / 2), angle, 1)
                bcode_roi = cv2.warpAffine(gray_img[bRect[1]:bRect[1] + bRect[3], bRect[0]:bRect[0] + bRect[2]], rotation_M, (bRect[2], bRect[3]), None, cv2.INTER_CUBIC) \
                            [roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2]]
                
                out_bounding_rect.append(rect)
                out_bcode_imgs.append(bcode_roi)
                out_shocked_bcode_imgs.append(shock_filter(bcode_roi, self.shock_sigma, self.shock_str_sigma, self.shock_blend, self.shock_iter, self.shock_thresh)) 
                
        #calculate center of gravity
        out_len = len(out_bounding_rect)
        x = y = 0
        w, h = img.shape[:2]
        center = (h / 2, w / 2)
        for rect in out_bounding_rect:
            x += rect[0][0]
            y += rect[0][1]
            
        if out_len > 0: center = (x / out_len, y / out_len)
        
        return (out_bounding_rect, out_bcode_imgs, out_shocked_bcode_imgs, center)
    
    def decode(self, img):
        zImg = zbar.Image(img.shape[1], img.shape[0], 'Y800', img.tostring())
        self.scanner.scan(zImg)
        
        for symbol in zImg: return str(symbol.data)
        return ''
        
    def match(self, imgs1, imgs2):
        matches = [None for i in range(len(imgs1))]
        imgs2_copy = list(imgs2)
        for i, img1 in enumerate(imgs1):
            scores = []
            for img2 in imgs2_copy:
                h1, w1 = img1.shape[:]
                h2, w2 = img2.shape[:]
                
                # cropping imgs to align dimensions
                w_to_pad = w1 - w2
                h_to_pad = h1 - h2
                if abs(h_to_pad) < 10:
                    if w_to_pad < 0:
                        pad_w = w1 + abs(w_to_pad)
                        pad_h = h1
                        w_pad_img = img1
                        other_img = img2
                    else: 
                        pad_w = w2 + w_to_pad
                        pad_h = h2
                        w_pad_img = img2
                        other_img = img1
                    
                    diff_x = pad_w - w_pad_img.shape[1]
                    start_x = math.floor(diff_x / 2.0)
                    w_cropped_img = other_img[:, start_x:pad_w - (start_x + diff_x % 2)]
                    
                    h1, w1 = w_pad_img.shape[:]
                    h2, w2 = w_cropped_img.shape[:]
                    h_to_pad = h1 - h2 
                    if h_to_pad < 0:
                        pad_w = w1
                        pad_h = h1 + abs(h_to_pad)
                        h_pad_img = w_pad_img
                        other_img = w_cropped_img 
                    else:
                        pad_w = w2
                        pad_h = h2 + h_to_pad
                        h_pad_img = w_cropped_img
                        other_img = w_pad_img    
                    
                    diff_y = pad_h - h_pad_img.shape[0]
                    start_y = math.floor(diff_y / 2.0)
                    h_cropped_img = other_img[start_y:pad_h - (start_y + diff_y % 2), :]
                    
                    result_img = cv2.absdiff(h_pad_img, h_cropped_img )
                    h, w = h_cropped_img.shape[:]
                    scores.append(np.sum(result_img) / (h * w))
                else:
                    scores.append(sys.maxint)
                    
            if len(scores) > 0:
                match_idx = np.argmin(scores)
                if scores[match_idx] < 20000:
                    matches[i] = imgs2_copy[match_idx]
                    del imgs2_copy[match_idx]
        
        return matches
        
    def get_shock_sigma(self):
        return self.shock_sigma


    def get_shock_str_sigma(self):
        return self.shock_str_sigma


    def get_shock_blend(self):
        return self.shock_blend


    def get_shock_iter(self):
        return self.shock_iter


    def get_shock_thresh(self):
        return self.shock_thresh


    def set_shock_sigma(self, value):
        self.shock_sigma = value if value % 2 == 1 else self.shock_sigma


    def set_shock_str_sigma(self, value):
        self.shock_str_sigma = value if value % 2 == 1 else self.shock_str_sigma


    def set_shock_blend(self, value):
        self.shock_blend = value


    def set_shock_iter(self, value):
        self.shock_iter = value


    def set_shock_thresh(self, value):
        self.shock_thresh = value
        
        
        
class BarcodeInfo(object):
    
    def __init__(self, bcode_img, decode_str = '', barcode_frame, contour):
        self.ID = id(bcode_img)
        self.bcode_img = bcode_img
        self.decode_str = decode_str
        self.contour = contour
        self.barcode_frame = barcode_frame  
        
        self.color = (randint(0, 255), randint(0, 255), randint(0, 255))
        
        self.update_candidates = []
        
        self.to_remove = False 
        
    def add_update_candidate(self, update_candidate):
        self.update_candidates.append(object)
        
    def update(self):
        new_barcodes = []
        if len(self.update_candidates) == 0:
            self.to_remove = True
        else:
            update_obj = min(self.update_candidates)[1]
            if self.decode_str == '':
                self.decode_str = update_obj.decode_str
            self.contour = update_obj.contour
            self.bcode_img = update_obj.bcode_img
            self.update_candidates = []
            
            for b_frame in self.update_candidates:
                if b_frame != update_obj:
                    new_barcodes.append(b_frame)
        
        return new_barcodes

class BarcodeFrame(object):
    
    def __init__(self, x, y, alpha, bcode_img, decode_str, contour):
        
        self.bcode_img = bcode_img
        self.decode_str = decode_str
        self.contour = contour
        
        self.x = x
        self.y = y
        self.alpha = -alpha
        self.context = []
        self.rotation_M = np.array((cos(self.alpha),  sin(self.alpha), -self.x), 
                                   (-sin(self.alpha), cos(self.alpha), -self.y), 
                                   (0               , 0              ,  1))
        
    def add_context_point(self, x, y):
        p = np.array((x, y, 1))
        #normalizzo l'angolo del punto di contesto rispetto all'angolo del punto corrente 
        rotated_p = self.rotation_M.dot(p)
        #aggiungo le coordinate polari del punto di contesto
        self.context.append(np.array((np.linalg.norm(rotated_p), math.atan2(rotated_p[1], rotated_p[0]))))
        #riordino la lista del contesto per angolo di rotazione 
        self.context.sort(key=itemgetter(1, 0)) 
        
    def match_score(self, barcode_frame):
        score = 0
        for p in barcode_frame.context:
            score += np.linalg.norm(p)
        return score