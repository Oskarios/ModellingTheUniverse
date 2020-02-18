# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 21:50:06 2020

@author: Oskar
"""

import numpy as np
from vpython import *


array1 = np.array(vector(1,2,3))
array2 = np.array(vector(4,5,6))
array3 = np.array([1,2,3])
array4 = np.array([4,5,6])

print(np.stack((array1,array2),0))
print(np.append(array1,array2))
print(np.append(array3,array4))
print(np.stack((array3,array4),0))

progress = np.stack((array3,array4),0)

print(progress)
print(np.vstack((progress,array4)))
print(np.vstack((progress,array4)).shape[0])

arr = np.array([[0,1],[2,3]])
print(arr[0][1])