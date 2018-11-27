#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 10:22:37 2018

@author: inti.pedroso
"""

import numpy as np
import schwimmbad
import pandas as pd

from tqdm import tqdm
from scipy.stats import pareto, gamma, beta
from scipy.special import gammaln, expit
from scipy import logaddexp
from conflation import beta_conflation


def _log_beta_binomial_density(k,n,alpha,beta):
    uno = gammaln(n+1) - (gammaln(k+1) + gammaln(n-k+1))
    dos = gammaln(k+alpha) + gammaln(n-k+beta) - gammaln(n+alpha+beta)
    tres = gammaln(alpha + beta) - (gammaln(alpha) + gammaln(beta))
    return uno + dos + tres

def log_beta_binomial_loop(counts, pars):
    totals = np.sum(counts,1)
    return np.apply_along_axis(lambda x: _log_beta_binomial_density(counts[:,0],totals, x[0],sum(x)), 1, pars)

def get_mixture_membership(data, pars, log = True):
    w = log_beta_binomial_loop(data, pars )
    w = w - logaddexp.reduce(w,axis=0)
    if not log:
        w = exp_(w)
    return w


def lnprob(x, means, local_data, count_tuple_frequency):

    #local_data = other_args[0]
    llike = lnlike(x,local_data, means, count_frq = count_tuple_frequency, return_pi = True)
    lprior = lnprior(x, pi=llike['pi'])
    log_prob = llike['ll'] + lprior
    #print log_prob, xn
    return log_prob

def lnprior(x, pi=None, local_CRPpar=10.0):
    back = pareto.logpdf(x=x, b=1.5, scale=10).sum()
    if pi is not None:
        back += beta.logpdf(pi,1,local_CRPpar).sum() 
    return back

def lnlike(x,local_data, means, count_frq = None, return_pi = False, return_logz = False):
    local_pars = np.array([means * x, (1.0 - means)*x]).T
    ll = log_beta_binomial_loop(local_data, local_pars )
    log_z = ll - logaddexp.reduce(ll,axis=0)
    if count_frq is not None:
        log_z = log_z * count_frq
    pi_ = logaddexp.reduce(log_z,axis=1)
    pi = exp_(pi_ - logaddexp.reduce(pi_))
    back = {}
    back['pi'] = None
    back['log_z'] = None
    if return_pi:
        back['pi'] = pi
    if return_logz:
        back['log_z'] = log_z
    if count_frq is not None:
        back['ll'] = np.dot(np.multiply(ll,count_frq).T,pi).sum()
    else:
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
    back += beta.logpdf(local_pi,1,local_CRPpar).sum()
    back += gamma.logpdf(local_CRPpar,4,3).sum()
    if np.isnan(back):
        back = -np.infty
    return back


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
            [exp_(np.arange(start,end,step)),
            0.5,
            1 - exp_(np.arange(start,end,step))]
        )
    )

    mus[mus == 0] = 0.01
    mus[mus == 1] = 0.99
    return mus

def get_mu_linear(K=3):
    back = np.sort(np.hstack([np.arange(0,1.0,1.0/(K-1))] + [.99]))
    back[back == 0] = 0.01
    back[back == 1] = 0.99
    return back

    
def get_prior_counts(K=3, center_prop=0.9):
    pc = np.ones(K)
    pc[int(0.5*(K-1))] = 10
    return pc


def get_observation_post( counts, prior_pars, weights=None, ncores=1,mpi=False, chunk=12):
    local_K = prior_pars.shape[0]
    if weights is None:
        weights = np.ones((local_K,))
    w = get_mixture_membership(counts, prior_pars, log=False) * weights.reshape((local_K,1))
    w = w/w.sum(0)
    pool = schwimmbad.choose_pool(mpi=mpi, processes=ncores)
    acc = 0
    total = counts.shape[0]
    pbar = tqdm(total=total)
    back = []
    for i in xrange( total/chunk):
        back.append( pool.map(beta_conflation, [ (local_c + prior_pars, local_w) for local_c, local_w in zip(counts[acc:acc+chunk,:],w.T) ] ) )
        acc += chunk
        pbar.update(chunk)
    
    back.append(pool.map(beta_conflation, [ (local_c + prior_pars, local_w) for local_c, local_w in zip(counts[acc:,:],w.T) ] ) )
    pbar.update(total - acc)    
    pool.close()
    back = pd.concat([ pd.DataFrame(df) for df in back]).values
    
    return back
    
