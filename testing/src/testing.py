# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.

__author__ = "mayurawijeyaratne"
__date__ = "$Sep 13, 2015 8:33:49 AM$"

import pandas as pd
import numpy as np
import scipy.spatial as scsp
from pandas import DataFrame
from sklearn.cluster import KMeans
from sklearn import preprocessing

def leastSq(x,y):
        A = np.vstack([x, np.ones(len(x))]).T
        m, c = np.linalg.lstsq(A, y)[0]
        return m
    
def gradEyes(x,y,x1,y1):
    leftEye = leastSq(x,y)
    rightEye = leastSq(x1,y1)
    return (leftEye-rightEye)/2

def normalizeFactor(point27, point28):
    norFac = scsp.distance.cdist(point27, point28, 'euclidean')
    return norFac

def distEyesEyebrows(points, lN):
    totalEucDis = 0
    for i in range(len(points)-10):
        eucDis = scsp.distance.cdist(points[i], points[i+10], 'euclidean')
        totalEucDis += eucDis
    distEyes = totalEucDis/(10*lN)
    return distEyes

def areaOfTriangle(point1, point2, point3):
    area = abs((point1[0][0]*(point2[0][1]-point3[0][1]) + point2[0][0]*(point3[0][1]-point1[0][1]) + point3[0][0]*(point1[0][1]-point2[0][1])) / 2)
    return area
    
def areaBetweenEyes(point5, point6, point16, point15, lN):
    tri1 = areaOfTriangle(point5,point6,point16)
    tri2 = areaOfTriangle(point5,point16,point15)
    areaEyes = (tri1 + tri2)/ lN**2
    return areaEyes

def areaofOctogon(point1,point2,point3,point4,point5,point6,point7,point8):
    tri1 = areaOfTriangle(point1,point2,point3)
    tri2 = areaOfTriangle(point1,point3,point4)
    tri3 = areaOfTriangle(point1,point4,point5)
    tri4 = areaOfTriangle(point1,point5,point6)
    tri5 = areaOfTriangle(point1,point6,point7)
    tri6 = areaOfTriangle(point1,point7,point8)
    area = tri1 + tri2 + tri3 + tri4 + tri5 + tri6
    return area

def areaOfEyes(point11,point12,point13,point14,point15,point23,point22,point21,point16,point17,point18,point19,point20,point26,point25,point24,lN):
    
    areaLeftEye = areaofOctogon(point11,point12,point13,point14,point15,point23,point22,point21)
    areaRightEye = areaofOctogon(point16,point17,point18,point19,point20,point26,point25,point24)
    areaEyesBoth = (areaLeftEye + areaRightEye) * (1/lN**2)
    return areaEyesBoth

def vTHRofEyes(point22,point13,point15,point11,point25,point18,point20,point16):
    distance1 = scsp.distance.cdist(point22, point13, 'euclidean')
    distance2 = scsp.distance.cdist(point15, point11, 'euclidean')
    distance3 = scsp.distance.cdist(point25, point18, 'euclidean')
    distance4 = scsp.distance.cdist(point20, point16, 'euclidean')
    arctan1 = np.arctan(distance1/distance2)
    arctan2 = np.arctan(distance3/distance4)
    feature5 = 0.5 * (arctan1 + arctan2)
#    print feature5
    return feature5

def areaCircumOfMouth(point1,point2,point3,point4,point5,point6,point7,point8,lN):
    area = areaofOctogon(point1,point2,point3,point4,point5,point6,point7,point8)
    feature6 = area / lN**2
    return feature6

def vTHRofCircMouth(p1,p2,p3,p4):
    distance1 = scsp.distance.cdist(p1, p2, 'euclidean')
    distance2 = scsp.distance.cdist(p3, p4, 'euclidean')
    f8 = np.arctan(distance1/distance2)
    return f8

def vposOfMouth(p29,p30,p31,p32,p33,p34,p35,p36,p37,p38,p39,p40,p41,p42,lN):
    a1 = np.array([p29[0][1],p30[0][1]])
    a2 = np.array([p31[0][1],p32[0][1],p33[0][1],p34[0][1],p35[0][1],p36[0][1],p37[0][1],p38[0][1],p39[0][1],p40[0][1],p41[0][1],p42[0][1]])
    f10 = (1/lN) * (np.mean(a1)-np.mean(a2))
    return f10
    

def main():
    print "Main Method"
    
    data_df = pd.read_csv('test.csv')
    df = DataFrame(data = data_df)
    
    no_of_frames = df.ix[df['FrameNumber'].idxmax()]
#    print no_of_frames.FrameNumber
    
    grouped = df.groupby('FrameNumber')
    grouped.groups
    
#    print grouped
#    print grouped.get_group(2)
    dfGroups = []
    
    keys = grouped.groups.keys()
    for i, val in enumerate(keys):
        dfGroups.append(grouped.get_group(val))
        dfGroup = grouped.get_group(val)
#        print dfGroup
    
#    print len(dfGroups[1])

#    points = []
    dfGroupPoints = []
    rows_list = []
    df1 = DataFrame(columns=('1', '2','3','4','5','6','7','8','9','10'))

    for index in range(len(dfGroups)):
#    for index in range(2):
        dfGroupPoints = []
        for row, frame in dfGroups[index].iterrows():
            dfGroupPoints.append([[frame.FeatureXAxis,frame.FeatureYAxis]])
        print "Index No= " + str(index) + "        Frame Number: " + str(frame.FrameNumber)
#        print dfGroupPoints
        point1 = dfGroupPoints[12]
        point2 = dfGroupPoints[18]
        point3 = dfGroupPoints[16]
        point4 = dfGroupPoints[19]
        point5 = dfGroupPoints[13]
        point6 = dfGroupPoints[14]
        point7 = dfGroupPoints[20]
        point8 = dfGroupPoints[17]
        point9 = dfGroupPoints[21]
        point10 = dfGroupPoints[15]
        point11 = dfGroupPoints[23]
        point12 = dfGroupPoints[35]
        point13 = dfGroupPoints[28]
        point14 = dfGroupPoints[36]
        point15 = dfGroupPoints[24]
        point16 = dfGroupPoints[25]
        point17 = dfGroupPoints[39]
        point18 = dfGroupPoints[32]
        point19 = dfGroupPoints[40]
        point20 = dfGroupPoints[26]
        point21 = dfGroupPoints[37]
        point22 = dfGroupPoints[27]
        point23 = dfGroupPoints[38]
        point24 = dfGroupPoints[41]
        point25 = dfGroupPoints[31]
        point26 = dfGroupPoints[42]
        point27 = dfGroupPoints[0]
        point28 = dfGroupPoints[1]
        point29 = dfGroupPoints[3]
        point30 = dfGroupPoints[4]
        point31 = dfGroupPoints[56]
        point32 = dfGroupPoints[54]
        point33 = dfGroupPoints[57]
        point34 = dfGroupPoints[59]
        point35 = dfGroupPoints[55]
        point36 = dfGroupPoints[58]
        point37 = dfGroupPoints[60]
        point38 = dfGroupPoints[61]
        point39 = dfGroupPoints[62]
        point40 = dfGroupPoints[65]
        point41 = dfGroupPoints[64]
        point42 = dfGroupPoints[63]
        
        x = [point12[0][0], point18[0][0], point16[0][0], point19[0][0], point13[0][0]]
        y = [point12[0][1], point18[0][1], point16[0][1], point19[0][1], point13[0][1]]

        x1 = [point14[0][0], point20[0][0], point17[0][0], point21[0][0], point15[0][0]]
        y1 = [point14[0][1], point20[0][1], point17[0][1], point21[0][1], point15[0][1]]

#        print "Frame Number = " + str(index) 
        
#        print "Grediant of eyebrows = " + str(feature1(x,y,x1,y1))

        lN = normalizeFactor(point27,point28)
        points1to20 = [point1,point2,point3,point4,point5,point6,point7,point8,point9,point10,point11,point12,point13,point14,point15,point16,point17,point18,point19,point20]

#        print "Normalizing Factor = " +  str(lN)
        
#        print "Distance between eyebrows and eyes = " + str(feature2(points1to20, lN))
    
#        print "Area betweens the eyes = " + str(areaBetweenEyes(point5,point6,point16,point17,lN))
#        print "Area of the eyes = " + str(areaOfEyes(point11,point12,point13,point14,point15,point23,point22,point21,point16,point17,point18,point19,point20,point26,point25,point24,lN))
#        print "Vertical to Horizonatal Ratio of Eyes = " + str(VTHRofEyes(point22,point13,point15,point11,point25,point18,point20,point16))
#        print "Area of the circumferance of the mouth = " + str(areaCircumOfMouth(point1,point2,point3,point4,point5,point6,point7,point8,lN))
#        print "Vertical to Horizontal ration of mouth = " + str(VTHRofCircMouth(point1,point2,point3,point4))
#        print "Vertical position of the corner of the mouth = " + str(VposOfMouth(point29,point30,point31,point32,point33,point34,point35,point36,point37,point38,point39,point40,point41,point42,lN))
    
        feature1 = gradEyes(x,y,x1,y1)
        feature2 = distEyesEyebrows(points1to20, lN)
        feature3 = areaBetweenEyes(point5,point6,point15,point16,lN)
        feature4 = areaOfEyes(point11,point12,point13,point14,point15,point23,point22,point21,point16,point17,point18,point19,point20,point26,point25,point24,lN)
        feature5 = vTHRofEyes(point22,point13,point15,point11,point25,point18,point20,point16)
        feature6 = areaCircumOfMouth(point29,point31,point32,point33,point30,point34,point35,point36,lN)
        feature7 = areaCircumOfMouth(point29,point37,point38,point39,point30,point40,point41,point42,lN)
        feature8 = vTHRofCircMouth(point35,point32,point30,point29)
        feature9 = vTHRofCircMouth(point41,point38,point30,point29)
        feature10 = vposOfMouth(point29,point30,point31,point32,point33,point34,point35,point36,point37,point38,point39,point40,point41,point42,lN)
#        feature11 = lN
        
        df1.loc[index] = [feature1,feature2,feature3,feature4,feature5,feature6,feature7,feature8,feature9,feature10]
        rows_list.append([feature1,feature2,feature3,feature4,feature5,feature6,feature7,feature8,feature9,feature10])
    
    rows_list_float = np.array(rows_list, dtype=float)

#    y_pred = KMeans(n_clusters=3).fit_predict(df1)
#    print y_pred

#    print y_pred.cluster_centers_ 

    a = 0
    x = 0
    y = 0
    z = 0
#    j = 0
#    for i in range(len(y_pred)):
#        if y_pred[i] == 0:
#            x += 1
#        if y_pred[i] == 1:
#            y += 1
##            print i
##            print df1.iloc[[i]]
#        if y_pred[i] == 2:
#            z += 1
##            print i
##            print df1.iloc[[i]]
#    print x
#    print y
#    print z
#    print df1

    min_max_scaler = preprocessing.MinMaxScaler()
    df1_normalize = min_max_scaler.fit_transform(df1)
    df1_new = df1_normalize[0:100]
    print df1_new
#    print df1_normalize
#    x_pred = min_max_scaler.fit_predict(df1)
    x_pred = KMeans(n_clusters=2).fit_predict(df1_new)
    
    for i in range(len(x_pred)):
        if x_pred[i] == 0:
            x += 1
#            print i
        if x_pred[i] == 1:
            y += 1
            print i
#            print df1.iloc[[i]]
        if x_pred[i] == 2:
            z += 1
        if x_pred[i] == 2:
            a += 1
#    print a
    print x
    print y
#    print z
    
    print "Finished"

if __name__ == "__main__":
    main()
