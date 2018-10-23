from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
import numpy as np
import csv
import random as rn
import urllib.request
import urllib.error
import math
import cv2 as cv
import pandas
import time
# fix random seed for reproducibility
np.random.seed(42)

# The below is necessary for starting core Python generated random numbers
# in a well-defined state.

rn.seed(12345)




from keras.models import load_model
def inp1(X,Y):
    
    url="https://api.darksky.net/forecast/d67f5cbba560136ea307952b60ff38fb/"+str(X)+","+str(Y)
    data = str(urllib.request.urlopen(url).read(100000))
    index=data.find("daily")
    data=data[data.find('"precipIntensityMax":',index)+21:]
    data=data[:data.find(",")]
    
    data=float(data)*24
    print(data)
    
    rain_limit=0.5

    downpour_limit=8
    
    
    flooding_limit=50
    continuousrain_limit=20
    if(data<rain_limit):
        data1=1
    if(data>rain_limit and data<downpour_limit):
        data1=2
    if(data>downpour_limit and data<continuousrain_limit):
        data1=3
    if(data>continuousrain_limit and data<flooding_limit):
        data1=4
    if(data>flooding_limit):
        data1=5
    

    return data
def inp(X,Y,rows,cols):
    dist_B_W_L = 111.320 #distance in km between latitudes and longitudes in the equator
    rng = 1 #distance in km of the area of which we need to find elevation
    N=X+((rng/2)/dist_B_W_L)
    if N>90:
        N=N%90-90
    S=X-((rng/2)/dist_B_W_L)
    if S>90:
        S=(S%-90)+90
    E=Y+((rng/2)/(dist_B_W_L*(math.cos(math.radians(X)))))
    if E>180:
        E=E%180-180
    W=Y-((rng/2)/(dist_B_W_L*(math.cos(math.radians(X)))))
    if W<180:
        W=(W%-180)+180
    heights = "sealevel"
    box=str(S)+","+str(W)+","+str(N)+","+str(E)
    BingMapsKey="AsRCr-5CGNNFnojPR7utwIayhfTOsDz5aeAQbq51t1mOG0PhNLgr20bMQYLvAt7G"
    url="http://dev.virtualearth.net/REST/v1/Elevation/Bounds?bounds="+ str(box)+"&rows="+str(rows)+"&cols="+str(cols)+"&heights="+str(heights)+"&key="+str(BingMapsKey)
    try:
        data = str(urllib.request.urlopen(url).read(1000))
    except urllib.error.URLError as e:
        print("Error: \n",e)
        exit()

    
    return data

def getElevation(data,rows,cols):
    data=data[data.find('"elevations":'):]
    data=data[data.find('[')+1:data.find(']')]
    data=np.fromstring(data,dtype=int,sep=',')
    data = np.reshape(data,(rows,cols))
    return data

def recur(data,r,c,X,Y,sum,k):
    maxI=r
    maxJ=c

    dist_B_W_L = 111.320 #distance in km between latitudes and longitudes in the equator
    rng = 1 #distance in km of the area of which we need to find elevation
    Lat_Diff=((rng/2)/dist_B_W_L)
    Long_Diff=((rng/2)/(dist_B_W_L*(math.cos(math.radians(X)))))

    for i in range(-1,2):
        for j in range(-1,2):
            try:
                if(data[maxI,maxJ]<=data[r+i,c+j]):
                    maxI=r+i
                    maxJ=c+j
            except Exception as e:
                print(e)
                continue
    pv=pixelValue(X+((maxI-5)/4)*Lat_Diff,Y+((maxJ-5)/4)*Long_Diff)
    if(pv>=0):
        k=k+1
        sum=sum+pv
    if(maxI==r and maxJ==c):
        return [maxI,maxJ,sum/k]
    return recur(data,maxI,maxJ,X,Y,sum,k)

def pixelValue(X,Y):
    XI=(math.floor(X/10)*10)
    YI=(math.floor(Y/10)*10)
    if(XI<0):
        XS=str(abs(XI))+"S"
    else:
        XS=str(XI)+"N"
    if(YI<0):
        YS=str(abs(YI))+"W"
    else:
        YS=str(YI)+"E"
    try:
        Mat = cv.imread("LC_Data/SLC_"+XS+"_"+YS+".tif",0)
        return Mat[int(400*(X-XI))][int(400*(Y-YI))]
    except Exception as e:
        print("Error in finding pixel Value: \n",e)
        return -1

def slope(data,rows,cols,X,Y):
    [Mi,Mj]=[int((rows +1)/2),int((cols+1)/2)]
    [Ti,Tj,avg]=recur(data,int((rows +1)/2),int((cols+1)/2),X,Y,pixelValue(X,Y),1)
    dist=(math.sqrt(math.pow(Ti-Mi,2)+math.pow(Tj-Mj,2))/9)
    alt=abs(data[Ti][Tj]-data[Mi][Mj])
    return [float(alt)/float(dist),avg]


def populate(X,Y):
    X=float(X)#Latitude
    Y=float(Y)#Longitude
    rows = 9
    cols = 9
    data=inp(X,Y,rows,cols)
    E=getElevation(data,rows,cols)

    data1=inp1(X,Y)
    data2=slope(E,rows,cols,X,Y)
    X1=np.array([data,data2,data1])
    pred_diff=np.zeros((1,6) ,dtype=float)
    for c in range(1,6):
        X2=np.array([data,data2,c])
        pred_diff[c-1]=nn(X2)
    pred_diff[5]=nn(X1)

    return pred_diff#Call this to get the landslide estimation in all weather and the last index is the current landslide prediction


def nn(X1):
    new_model=load_model('my_model.h5')



    predictions = new_model.predict(X1)
    # round predictions
    rounded = np.array([round(X1[0]) for X1 in predictions])

    return rounded# round predictionsc"""
