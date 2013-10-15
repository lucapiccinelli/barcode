# -*- coding: utf-8 -*-
"""
Created on Wed Sep 04 16:18:38 2013

@author: Luca
"""

from BarcodeTools import *
import matplotlib.pyplot as plt
import cv2
import os

def plotFeatures(f1, f2):
    score = 0
    for p1, p2 in zip(f1, f2):
        diff = (p1[0] - p2[0], (p1[1] - p2[1] + 180) % 360 - 180)
        score += np.linalg.norm(diff)
    
    plt.plot([p[0] for p in f1], [p[1] for p in f1], 'go')
    plt.plot([p[0] for p in f2], [p[1] for p in f2], 'yo')
    for i, p in enumerate(f1): plt.text(p[0], p[1] + 5, str(i), bbox={'facecolor':'green',  'alpha':0.5})
    for i, p in enumerate(f2): plt.text(p[0], p[1] - 5, str(i), bbox={'facecolor':'yellow', 'alpha':0.5})
    plt.title(score)
    plt.grid(True)
    plt.show()

def normalize(ref_point, points):
    RADIANS_SCALE = 0.017453292519943295
    alpha = (ref_point[2] if ref_point[2] > -45 else ref_point[2] + 90)  * RADIANS_SCALE# math.pi / 180.0
    x = ref_point[0][0]
    y = ref_point[0][1]
            
    rotation_M = np.array(([cos(alpha),  sin(alpha), 0.0], 
                           [-sin(alpha), cos(alpha), 0.0], 
                           [0.0        , 0.0       , 1.0]))
    translation2_M = np.array(([1.0, 0.0, -x],
                               [0.0, 1.0, -y],
                               [0.0, 0.0, 1.0]))
    transform_M = rotation_M.dot(translation2_M)
    
    normalized_points = [np.around(transform_M.dot(np.array((p[0][0], p[0][1], 1))), decimals=4) for p in points]
    transformed_points = []
    for p in normalized_points:
        tp = (np.linalg.norm(np.array((p[0], p[1]))), math.atan2(p[1], p[0]) / RADIANS_SCALE)
        transformed_points.append(np.array((tp[0], tp[1] if tp[1] >= 0 else tp[1] + 360)))
    
    return transformed_points

colors = [(randint(0, 255), randint(0, 255), randint(0, 255)) for i in range(10)]

path_dir = 'D:\\tmp\\frames'
filname_list = os.listdir(path_dir)

fr1 = cv2.imread('\\'.join([path_dir, filname_list[0]]))
for i in range(len(filname_list)):
    if i > 0:
        f_name2 = filname_list[i]

        fr2 = cv2.imread('\\'.join([path_dir, f_name2]))
        
        tool = BarcodeTools(True)
        rects,  codes,  shocked,  bla  = tool.detect(fr1)
        rects2, codes2, shocked2, bla2 = tool.detect(fr2)
        
        if len(rects) == 0 or len(rects2) == 0: continue
        
        out_points1 = [normalize(r, rects) for r in rects]
        out_points2 = [normalize(r, rects2) for r in rects2]
        
        #for op in out_points1: op.sort(key=itemgetter(1, 0))
        #for op in out_points2: op.sort(key=itemgetter(1, 0))
        
        #plotFeatures(out_points1[4], out_points2[5])
        
        couples = []
        QUANT_FACTOR = 1.0 / 50.0
        max_angle = int(360 * QUANT_FACTOR) + 1
        max_length = int(np.linalg.norm(fr1.shape) * QUANT_FACTOR) + 1
        mat1 = np.zeros((max_angle, max_length))
        mat2 = np.zeros((max_angle, max_length))
        
        for i, op2 in enumerate(out_points2):
            best_score = [0, sys.maxint]
            mat2.fill(0)
            for p2 in op2: mat2[floor(p2[1] * QUANT_FACTOR)][floor(p2[0] * QUANT_FACTOR)] += 1
            print ' '
            for j, op1 in enumerate(out_points1):
                score = 0
                mat1.fill(0)
                for p1 in op1: mat1[floor(p1[1] * QUANT_FACTOR)][floor(p1[0] * QUANT_FACTOR)] += 1
        #        score = -cv2.GaussianBlur(mat2 + mat1, (3, 3), 1).sum()
                score = absolute((mat2 - mat1)).sum()
                    
                if score < best_score[1]:
                    best_score[1] = score
                    best_score[0] = j
                
            couples.append((i, best_score[0], best_score[1]))
        
        #for i, op2 in enumerate(out_points2):
        #    best_score = [0, sys.maxint]
        #    for j, op1 in enumerate(out_points1):
        #        score = 0
        #        for p1, p2 in zip(op1, op2):
        #            diff = (p1[0] - p2[0], (p1[1] - p2[1] + 180) % 360 - 180)
        #            score += np.linalg.norm(diff)
        #            
        #        if score < best_score[1]:
        #            best_score[1] = score
        #            best_score[0] = j
        #        
        #    couples.append((i, best_score[0], best_score[1]))
            
        print couples
        
        #plotting
        #points1 = [(rect[0][0], rect[0][1]) for rect in rects]
        #points2 = [(rect[0][0], rect[0][1]) for rect in rects2]
        #plt.plot([p[0] for p in points1], [p[1] for p in points1], 'ro')
        #plt.plot([p[0] for p in points2], [p[1] for p in points2], 'bo')
        #plt.axis([-10, 700, -10, 500])
        #plt.grid(True)
        #plt.show()
        
        for i, rect in enumerate(rects):    
            box = np.int0(cv2.cv.BoxPoints(rect))
            cv2.drawContours(fr1, [box], 0, colors[i], thickness = 2)
            cv2.putText(fr1, '{2} - {0}, {1}'.format(rect[0][0], rect[0][1], i), rect[0], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), thickness=1)
            
            
#        for i, rect in enumerate(rects2):
#            box = np.int0(cv2.cv.BoxPoints(rect))
#            cv2.drawContours(fr2, [box], 0, colors[couples[i][1]], thickness = 2)    
#            cv2.putText(fr2, '{2} - {0}, {1}'.format(rect[0][0], rect[0][1], i), rect[0], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), thickness=1)
        
        cv2.imshow('blo', fr1)
        #cv2.imshow('blo2', fr2)

        fr1 = fr2
        
#        while True:
        k =  cv2.waitKey(30)
        if k == 27: break
        
        
cv2.destroyAllWindows()