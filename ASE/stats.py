#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 16:54:06 2018

@author: inti.pedroso
"""

from scipy.special import betaln as lbeta
import numpy as np
from utils import exp_

def prob_2_beats_1_parameters(alpha_1, beta_1, alpha_2, beta_2):
    total = 0.0
    for i in np.arange(0,alpha_2-1):
        total += exp_( lbeta(alpha_1+i,beta_1 + beta_2) - np.log(beta_2+i) - lbeta(1+i,beta_2) - lbeta(alpha_1,beta_1))
    return total

def prob_1_diff_2(alpha_1, beta_1, alpha_2, beta_2):
    p = prob_2_beats_1_parameters(alpha_1, beta_1, alpha_2, beta_2)
    return np.abs(p - (1-p))

def prob_ASE_mixt_prior(alpha,beta,pars):
    local_K = pars.shape[0]
    null_pars = pars[(local_K-1)/2,:]
    prior_w = 1.0 - np.array([ prob_1_diff_2(pars[i,0],pars[i,1],null_pars[0],null_pars[1]) for i in xrange((local_K -1)/2 + 1)])
    prob_ASE = np.array([ prob_1_diff_2(alpha,beta,pars[i,0],pars[i,1]) for i in xrange((local_K -1)/2 + 1)])
    mixt_prob_ASE = np.dot(prob_ASE, prior_w/prior_w.sum())
    return mixt_prob_ASE