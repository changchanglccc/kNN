#kNN classification
from numpy import *
import operator
from os import listdir

def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group,labels

#calculate distance
def classify0(inX,dataSet,labels,k): #inX:input vector; dataSet: traning data; labels: training label; k neigbours
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX,(dataSetSize,1)) - dataSet #tile():repeat array inX, "dataSet" rows, repeat once;calculate the subtraction value.
    sqDiffMat = diffMat ** 2
    sqDistance = sqDiffMat.sum(axis=1)
    distance = sqDistance ** 0.5
    sortedDist = distance.argsort()

    #选择距离最小的k个点
    classCount={}
    for i in range(k):
        voteIlabel=labels[sortedDist[i]] #根据排序结果的索引值返回靠近的前k个标签
        classCount[voteIlabel]=classCount.get(voteIlabel,0)+1 #各个标签出现频率
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True) #排序频率
    return sortedClassCount[0][0]  #找出频率最高的

def imgToVector(filename):
    myVector = zeros((1,1024)) #现在是一维的向量
    file = open(filename)
    for i in range(32):
        line = file.readline()
        for j in range(32):
            myVector[0,32*i+j]=int(line[j]) #把二维的file，输入到一维的向量里面
    return myVector

#test classifier
