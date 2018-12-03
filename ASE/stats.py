#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 16:54:06 2018

@author: inti.pedroso
"""

from scipy.special import betaln as lbeta
from scipy import logaddexp
import numpy as np
from utils import exp_

def prob_2_beats_1_binary(alpha_1, beta_1, alpha_2, beta_2, invert_to_seep_up=True, pseucount=1):
    if invert_to_seep_up and ( alpha_2 > alpha_1):
        alpha_1, alpha_2 = (alpha_2,alpha_1)
        beta_1, beta_2 = (beta_2,beta_1)

    alpha_1 += pseucount
    beta_1 += pseucount
    alpha_2 += pseucount
    beta_2 += pseucount        
    
    total = list()
    for i in xrange(0,int(np.around(alpha_2-1))):
        total.append( lbeta(alpha_1+i,beta_1 + beta_2) - np.log(beta_2+i) - lbeta(1+i,beta_2) - lbeta(alpha_1,beta_1))
    total = exp_(logaddexp.reduce(total))
    return total

def prob_2_beats_1_counts(alpha_1, beta_1, alpha_2, beta_2, invert_to_seep_up=True, pseucount=1):
    if invert_to_seep_up and (alpha_2 < alpha_1):
        alpha_1, alpha_2 = (alpha_2,alpha_1)
        beta_1, beta_2 = (beta_2,beta_1)
   
    alpha_1 += pseucount
    beta_1 += pseucount
    alpha_2 += pseucount
    beta_2 += pseucount
    
    total = list()
    for k in xrange(0,int(np.around(alpha_1-1))):
        total.append(k*np.log(beta_1) + alpha_2*np.log(beta_2) - (k+alpha_2)*np.log(beta_1 + beta_2) - np.log(k+alpha_2) - lbeta(k+1,alpha_2))
    total = exp_(logaddexp.reduce(total))
    return total

def prob_1_diff_2(alpha_1, beta_1, alpha_2, beta_2, p_type="counts",invert_to_seep_up=True, pseucount=1):
    p = np.nan
    
    if p_type == "counts":
        p = prob_2_beats_1_counts(alpha_1, beta_1, alpha_2, beta_2,invert_to_seep_up=invert_to_seep_up, pseucount=pseucount)
    elif p_type == "binary":
        p = prob_2_beats_1_binary(alpha_1, beta_1, alpha_2, beta_2,invert_to_seep_up=invert_to_seep_up, pseucount=pseucount)
    
    else:
        print "Only probability types for binary and counts are allowed"
    return np.abs(p - (1-p))

def prob_ASE_mixt_prior(alpha,beta,pars, null_pars=None, prior_w = None,invert_to_seep_up=True, pseucount=1):
    local_K = pars.shape[0]
    if null_pars is None:
        null_pars = pars[(local_K-1)/2,:]
    if prior_w is None:
        prior_w = 1.0 - np.array([ prob_1_diff_2(pars[i,0],pars[i,1],null_pars[0],null_pars[1],invert_to_seep_up=invert_to_seep_up, pseucount=pseucount) for i in xrange(local_K)])
    prior_w /= prior_w.sum()
    prob_ASE = np.array([ prob_1_diff_2(alpha,beta,pars[i,0],pars[i,1],invert_to_seep_up=invert_to_seep_up, pseucount=pseucount) for i in xrange(local_K)])
    mixt_prob_ASE = np.dot(prob_ASE, prior_w)
    return mixt_prob_ASE