#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 10:22:37 2018

@author: inti.pedroso
"""

from scipy.stats import pareto, gamma, beta
from scipy.special import gammaln
from scipy import logaddexp

import numpy as np

from utils import exp_


def _log_beta_binomial_density(k,n,alpha,beta):
    uno = gammaln(n+1) - (gammaln(k+1) + gammaln(n-k+1))
    dos = gammaln(k+alpha) + gammaln(n-k+beta) - gammaln(n+alpha+beta)
    tres = gammaln(alpha + beta) - (gammaln(alpha) + gammaln(beta))
    return uno + dos + tres

def lnprob(x, means, local_data):
    #local_data = other_args[0]
    llike = lnlike(x,local_data, means, return_pi = True)
    lprior = lnprior(x, pi=llike['pi'])
    log_prob = llike['ll'] + lprior
    #print log_prob, x
    return log_prob

def lnprior(x, pi=None, local_CRPpar=10.0):
    back = pareto.logpdf(x=x, b=1.5, scale=10).sum()
    if pi is not None:
        back += beta.logpdf(pi,1,local_CRPpar).sum() 
    return back

def lnlike(x,local_data, means, return_pi = False, return_logz = False):
    ll = np.vstack([ _log_beta_binomial_density(local_data[:,0],
                                                local_data[:,1],
                                                alpha=mu*m,
                                                beta=(1.0-mu)*m) 
                    for mu,m in zip(means,x)])
    log_z = ll - logaddexp.reduce(ll,axis=0)
    pi_ = logaddexp.reduce(log_z,axis=1)
    pi = exp_(pi_ - logaddexp.reduce(pi_))
    back = {}
    back['pi'] = None
    back['log_z'] = None
    if return_pi:
        back['pi'] = pi
    if return_logz:
        back['log_z'] = log_z 
    back['ll'] = np.dot(ll.T,pi).sum()
    return back


def lnprob_mixprop(x, means, local_K):
    #local_data = other_args[0]
    local_M = x[:local_K]
    local_pi = zs_to_alphas(x[local_K:])

    llike = lnlike_mixprop(local_M, local_pi, local_K, this_data, means, return_pi = True)
    lprior = lnprior_mixprop(local_M, local_pi)
    log_prob = llike['ll'] + lprior
    #print log_prob, x
    return log_prob

def lnlike_mixprop(local_M,local_pi, local_K, local_data, means, return_pi = False, return_logz = False):
    ll = np.vstack([ _log_beta_binomial_density(local_data[:,0],
                                                local_data[:,1],
                                                alpha=mu*m,
                                                beta=(1.0-mu)*m) 
                    for mu,m in zip(means,local_M)])
    log_z = ll - logaddexp.reduce(ll,axis=0)
    back = {}
    back['log_z'] = None
    if return_logz:
        back['log_z'] = log_z 
    back['ll'] = np.dot(ll.T,local_pi).sum()
    return back

def lnprior_mixprop(local_M, local_pi, sbp_prior=6):
    back = 0.0
    back += beta.logpdf(local_pi,1,sbp_prior).sum()
    back += pareto.logpdf(x=local_M, b=1.5, scale=10).sum()
    return back


def lnprob_CRP(x, means, local_data):
    local_K = len(means)
    local_M = x[:local_K]
    local_CRPpar = x[local_K]
    
    llike = lnlike(local_M,local_data, means, return_pi = True)
    lprior = lnprior_CRP(local_M, local_pi=llike['pi'], local_CRPpar=local_CRPpar)
    #print llike['ll'], lprior, x
    log_prob = llike['ll'] + lprior
    return log_prob

def lnprior_CRP(x, local_pi, local_CRPpar):
    back = 0.0
    back += pareto.logpdf(x=x, b=1.5, scale=10).sum()
    p#rint local_CRPpar, local_pi, beta.logpdf(local_pi,1,local_CRPpar)
    back += beta.logpdf(local_pi,1,local_CRPpar).sum()
    back += gamma.logpdf(local_CRPpar,4,3).sum()
    if np.isnan(back):
        back = -np.infty
    return back
