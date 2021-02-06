# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 15:05:57 2020

@author: Owner
"""

import numpy as np 
import math
import matplotlib.pyplot as plt
import scipy.stats as stats


def Gaussian_proposal(current, sd):
    proposal = np.random.normal(current, sd)
    return proposal
def acceptance(current, proposed, f, **kwargs):
    try:
        p = min(f(proposed, **kwargs)/f(current, **kwargs), 1)
        return p
    except:
        return 0
def Normal(x, mu, sigma):
    return math.pow(math.e, -(1/2)*math.pow((x-mu)/sigma,2))
def Beta(x, a, b):
    if x < 1 and x > 0:
        return math.pow(x, a-1) * math.pow(1-x, b-1)
    else: 
        return 0
def BetaSample(a = 1, b = 1, N = 10000): 
    oof = np.zeros(N)
    Done = False
    current = .5
    oof[0] = current
    i = 1
    while not Done:
        proposal = Gaussian_proposal(current, .05)
        p = acceptance(current, proposal, Beta, a = a, b = b)
        u = np.random.uniform()
        if u <= p:
            current = proposal
            oof[i] = current
            i += 1
        if oof[-1] != 0:
            Done = True
    x = np.linspace(0, 1, 100)
    plt.hist(oof, density = True)
    plt.plot(x, stats.beta.pdf(x, a, b))
    plt.show()
def GaussianSample(mu = 0, sigma =1, N = 10000):  
    oof = np.zeros(N)
    Done = False
    current = 0
    oof[0] = current
    i = 1
    while not Done:
        proposal = Gaussian_proposal(current, 1)
        p = acceptance(current, proposal, Normal, mu = mu, sigma = sigma)
        u = np.random.uniform()
        if u <= p:
            current = proposal
            oof[i] = current
            i += 1
        if oof[-1] != 0:
            Done = True
    x = np.linspace(-3.5, 3.5, 100)
    plt.hist(oof, density = True)
    plt.plot(x, stats.norm.pdf(x))
    plt.show()
BetaSample(2,5)
GaussianSample()
#steps: 
#1. Initialize a point
#2. Generate a sample using the distribution g(x) -> normal?
#3. Calculate f(x) for the proposed sample
#4. Calculate the acceptance probability
#5. Accept or reject sample