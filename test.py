# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 21:50:06 2020

@author: Oskar
"""

import numpy as np
from vpython import *

'''
'''
array1 = np.array(vector(1,2,3))
array2 = np.array(vector(4,5,6))
'''
array3 = np.array([1,2,3])
array4 = np.array([4,5,6])
'''
print(np.stack((array1,array2),0))
'''
print(np.append(array1,array2))
print(np.append(array3,array4))
print(np.stack((array3,array4),0))

progress = np.stack((array3,array4),0)

print(progress)
print(np.vstack((progress,array4)))
print(np.vstack((progress,array4)).shape[0])

arr = np.array([[0,1],[2,3]])
print(arr[0][1])

arrr = np.array([0,2,3,4])
arry = np.array([1,3,1,4])

print(np.stack((arrr,arry),1))


print(np.append(np.array([]),1))
'''
'''
A = np.array([0,1,2,3])
print(np.flip(A))
B = np.array([])
for i in range(0,A.size):
    for j in range(i+1,A.size):
        element = np.array([i,j])
        print(element)
        np.vstack((B,[element]))
print(B)
'''    
print([1,2])

print(np.array([1,2]))

f = np.array([[0,1],[2,3],[4,5]])
print(f)
print(np.flip(f,1))

print("Final")
print(np.vstack((f,np.flip(f,1))))





g = np.array([])
h = np.array([0,1,2,3,4,5])


a = np.array([[0,1,2],[3,4,5]])
print(a[:,-1])


#print(np.reshape(h,(2,3)))