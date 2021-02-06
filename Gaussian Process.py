# -*- coding: utf-8 -*-
"""
Created on Sun Aug 16 16:34:01 2020

@author: Owner
"""
import numpy as np
import statistics as stats
import scipy.stats as sci
import matplotlib.pyplot as plt
import math


def cov(x1, x2):
    # math.exp(-1.5 * (x1-x2)**2)
    return (1 + x1*x2)
def makeC(x):
    C = np.zeros((len(x),len(x)))
    for i in range(len(x)):
        for j in range(len(x)):
            C[i, j] = cov(x[i], x[j])
    return C
def makeR(x, x_star):
    R = np.zeros((len(x), len(x_star)))
    for i in range(len(x)):
        for j in range(len(x_star)):
            R[i, j] = cov(x[i], x_star[j])
    return R





A = np.loadtxt("synthdata2016.csv", delimiter = ",")
#global Year_mean
#Year_mean = stats.mean(A[:,0])
#global Year_sd
#Year_sd = stats.stdev(A[:,0])
#global time_sd 
#time_sd = stats.stdev(A[:, 1])
#global time_mean 
#time_mean = stats.mean(A[:, 1])
A = sci.zscore(A, axis = 0)
x = A[:, 0]
f = A[:, 1]
C = makeC(x) + np.identity(len(x))
print(np.linalg.det(C))
x_new =  np.linspace(min(x), max(x), 1000)
C_star = makeC(x_new) 
R = makeR(x, x_new)
c_inv = np.linalg.inv(C)
mu = np.dot(np.dot(R.T, c_inv), f)
Sigma = C_star - np.dot(np.dot(R.T, c_inv), R)
f_new = np.zeros((10, len(x_new)))
for i in range(10):
    f_new[i, :] = np.random.multivariate_normal(mu, Sigma)
plt.cla()
for i in range(1):
    plt.scatter(x_new, np.mean(f_new, axis = 0))
plt.scatter(x, f)
plt.show
