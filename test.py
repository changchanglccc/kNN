import numpy as np
import operator
from numpy import *

a=1
b=2
c=a+b
print (c)

b = np.array([6, 7, 8])
c = b.shape[0]
print(c)

a = np.sum([[0,1,2],[2,1,3]])
print(a)
a = np.sum([[0,1,2],[2,1,3]],axis=0)
print(a)
a = np.sum([[8,1,2],[2,1,3],[0,0,0]],axis=1)
print(a)
d = a.argsort() #直接给出排名的值[2 1 0](说明第一个行向量的和最大，排第三；最后一个行向量的和最小)
print(d)

classCount = {"A":6,"B":1,"C":0,"D":2}
sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True) #排序频率
print(sortedClassCount)

f= np.array([[5,8],
             [1,2]])
def imgToVector(filename):
    myVector = zeros((1,4))
    for i in range(2):
        for j in range(2):
            myVector[0,5*i+j]=int(f[j])
    return myVector

myVector = zeros((1,4))
print(myVector[0,0+1])




