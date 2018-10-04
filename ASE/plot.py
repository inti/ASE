#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 17:01:46 2018

@author: inti.pedroso
"""
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn') # pretty matplotlib plots
plt.rcParams['figure.figsize'] = (12, 8)
import scipy.stats as ss

def plot_normal(x_range, mu=0, sigma=1, cdf=False, **kwargs):
    '''
    Plots the normal distribution function for a given x range
    If mu and sigma are not provided, standard normal is plotted
    If cdf=True cumulative distribution is plotted
    Passes any keyword arguments to matplotlib plot function
    '''
    x = x_range
    if cdf:
        y = ss.norm.cdf(x, mu, sigma)
    else:
        y = ss.norm.pdf(x, mu, sigma)
    plt.plot(x, y, **kwargs)
    
def plot_beta(alpha=1, beta=1, n_points = 500, cdf=False, w = 1, color=None, normalise=True, **kwargs):
    '''
    Plots the normal distribution function for a given x range
    If mu and sigma are not provided, standard normal is plotted
    If cdf=True cumulative distribution is plotted
    Passes any keyword arguments to matplotlib plot function
    '''
    x = np.linspace(0, 1, n_points)
    if cdf:
        y = ss.beta.cdf(x, alpha, beta)
    else:
        y = w*ss.beta.pdf(x, alpha, beta)
    if normalise:
        y /= np.sum(y)
    plt.plot(x, y,color=color, **kwargs)



#for i, (mu,m) in enumerate(zip(means,post_M)):
#    print mu*m, (1.0-mu)*m
#    plot_beta(mu*m, (1.0-mu)*m, w=2.5*post_pi[i], normalise=False)
#plt.hist(this_data[:,0]/this_data[:,1], density=True, bins=50, color='lightgrey')

#_ = corner.corner(samples)