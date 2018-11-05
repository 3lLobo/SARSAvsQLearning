#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np 
import math
from scipy.stats import norm, uniform
import matplotlib.pyplot as plt

def expectation(n):
    est = 0
    for i in range(n):
        qsamp = np.random.uniform(-5, 5)
        fpq = qsamp**2 * (norm.pdf(qsamp, loc=0, scale=1)/ uniform.pdf(qsamp, loc=-5, scale=10))
        est += fpq
        
    return est/n


def expnorm(n):
    est = 0
    for i in range(n):
        samp = np.random.normal()
        est += samp**2
    
    return est/n

def expectation2(n):
    est = 0
    samples = []
    for i in range(n):
        qsamp = np.random.uniform(-1, 1)
        fpq = qsamp**2 * (p(qsamp)/ uniform.pdf(qsamp, loc=-1, scale=2))
        samples.append(fpq)
        est += fpq
    
    
    return est/n, samples

def SEM(n):
    mu, x = expectation2(n)
    sumsqer = 0
    for xi in x:
        sumsqer += (xi - mu)**2
    var = math.sqrt( sumsqer / (n-1))
    return var / math.sqrt(n)

def p(x):
    return (1 + math.cos(math.pi*x))/2



def plot():
    x, y = [], []

    for i in [10, 100, 1000, 10000]:
        for j in range(10):
            x.append(i)
            y.append(expectation(i))
            
    plt.plot(x,y, 'ro')
    plt.xscale('log') 
    plt.ylabel('Estimate')
    plt.xlabel('Number of samples')
    plt.savefig('samp.pdf', bbox_inches='tight')
    plt.show()
    
    return

def plot2():
    x, y = [], []

    for i in [10, 100, 1000, 10000]:
        for j in range(10):
            x.append(i)
            y.append(expectation2(i))
            
    plt.plot(x,y, 'ro')
    plt.xscale('log') 
    plt.ylabel('Estimate')
    plt.xlabel('Number of samples')
    plt.savefig('samp2.pdf', bbox_inches='tight')
    plt.show()
    
    return