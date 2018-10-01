#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 10:25:27 2018

@author: inti.pedroso
"""

import numpy as np
from scipy.special import expit 

def exp_(x):
    back = None
    try:
        back = np.exp(x)
    except OverflowError:
        back = expit(x)
    if back.any() == np.infty:
        back = expit(x)
    return back

def get_mus(K=3, center=0.5, denom_val = 2):
    mus = np.zeros(K)
    center_idx = int(0.5*(K-1))
    mus[ center_idx ] = 0.5
    denominator = np.sqrt(denom_val)
    for i in range(int(K - 0.5*(K-1)),K):
      mus[i] = mus[i-1]/denominator;
      mus[K - i - 1] = 1 - mus[i-1]/denominator;
        
    print(mus)

def get_mus_log(K=3, center=0.5):
    start = np.log(1)
    end=np.log(0.5)
    step= (end - start)/((K-1)/2)
    mus = np.sort(
        np.hstack(
            [np.exp(np.arange(start,end,step)),
            0.5,
            1 - np.exp(np.arange(start,end,step))]
        )
    )

    mus[mus == 0] = 0.01
    mus[mus == 1] = 0.99
    return mus

def get_mu_linear(K=3):
    back = np.sort(np.hstack([np.arange(0,1.0,1.0/(K-1))] + [.99]))
    back[back == 0] = 0.01
    back[back == 1] = 0.99
    #back[back == 0.5] = 0.53


    return back

    
def get_prior_counts(K=3, center_prop=0.9):
    pc = np.ones(K)
    pc[int(0.5*(K-1))] = 10
    return pc