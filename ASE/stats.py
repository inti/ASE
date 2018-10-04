#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 16:54:06 2018

@author: inti.pedroso
"""

from scipy.special import betaln as lbeta
import numpy as np

def prob_2_beats_1_parameters(alpha_1, beta_1, alpha_2, beta_2):
    total = 0.0
    for i in np.arange(0,alpha_2-1):
        total += np.exp( lbeta(alpha_1+i,beta_1 + beta_2) - np.log(beta_2+i) - lbeta(1+i,beta_2) - lbeta(alpha_1,beta_1))
    return total

def prob_1_diff_2(alpha_1, beta_1, alpha_2, beta_2):
    p = prob_2_beats_1_parameters(alpha_1, beta_1, alpha_2, beta_2)
    return np.abs(p - (1-p))