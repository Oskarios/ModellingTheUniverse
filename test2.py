# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 16:45:30 2020

@author: Oskar
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

a = np.array([-1,-2,-3,-3,-1,-3,-5,-1,-2,-1])
a = np.reshape(a,(5,2))
print(a)

values = np.mean(a,axis=1)
error = np.std(a,axis=1)/np.sqrt(a.shape[1])

print(values)
print(error)

print(np.array([i for i in range(0,10)]))

plt.errorbar(np.array([i for i in range(values.size)]),values,error,color='black',ls=' ', marker='x',capsize=5, capthick=1, ecolor='black')
plt.show()

print(np.vstack((values,error)))