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
from conflation import get_beta_cdf

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
    x = np.linspace(1e-8, 1-1e-8, n_points)
    if cdf:
        y = ss.beta.cdf(x, alpha, beta)
    else:
        y = w*ss.beta.pdf(x, alpha, beta)
    if normalise:
        y /= np.sum(y)
    plt.plot(x, y,color=color, **kwargs)


def plot_beta_mixture(pars,pi=None,n_points = 500, cdf=False, w = 1, color=None, normalise=True,components_means=None, plot_components = False,**kwargs):
    if pi is None:
        pi = np.ones((pars.shape[0],))
    cdf_matrix = np.vstack([ get_beta_cdf([p]) for p in pars])
    mixt_cdf = np.sum( cdf_matrix * pi.reshape((pars.shape[0],1)), 0)
    n_points = cdf_matrix.shape[1]
    x = np.linspace(1e-8, 1-1e-8, n_points)
    if normalise:
        mixt_cdf /= np.sum(mixt_cdf)
    plt.plot(x,mixt_cdf*w,color='black')
    
    if plot_components:
        for i in xrange(pars.shape[0]):
            p = components_means[i]
            M = np.sum(pars[i,:])
            plot_beta(p*M,(1-p)*M, w=pi[i]*w, normalise=True)

#for i, (mu,m) in enumerate(zip(means,post_M)):
#    print mu*m, (1.0-mu)*m
#    plot_beta(mu*m, (1.0-mu)*m, w=2.5*post_pi[i], normalise=False)
#plt.hist(this_data[:,0]/this_data[:,1], density=True, bins=50, color='lightgrey')

#_ = corner.corner(samples)
    
    