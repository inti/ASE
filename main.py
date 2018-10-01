#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 10:35:54 2018

@author: inti.pedroso
"""

from distributions import BetaBinomial, Mixt
from utils import get_mu_linear, exp_
import pandas as pd
import numpy as np


files = !ls /Users/inti.pedroso/DATA/ASE/phaser_gene/
files
data = pd.concat([ pd.read_table( "".join(["/Users/inti.pedroso/DATA/ASE/phaser_gene/",f])) for f in files ])


data.head()
data2 = data[data["totalCount"] > 10]
data2.shape


means = get_mu_linear(K=7)

mixt_comp = [ BetaBinomial(p=mu,M_0=2) for mu in means]

mixt_comp[len(mixt_comp)/2].alpha_0 = 1
mixt_comp[len(mixt_comp)/2].beta_0 = 1
mixt_comp[len(mixt_comp)/2].alpha = 1
mixt_comp[len(mixt_comp)/2].beta = 1
mixt_comp[len(mixt_comp)/2].M = 2

this_data = data2.loc[:,["aCount","totalCount"]].values.astype(float)
this_data = this_data[this_data[:,0] > 10, :] 
md = Mixt(component_dist=mixt_comp, data=this_data[:1000,:])
#print(md)

#for d in md.mixt_comp:
#    print d

md.fit(n_iter=10, print_delta=True, fit_by_eb = True, null_iter=3)


%pylab inline

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
    
def plot_beta(x_range, alpha=1, beta=1, cdf=False, w = 1, color=None, normalise=True, **kwargs):
    '''
    Plots the normal distribution function for a given x range
    If mu and sigma are not provided, standard normal is plotted
    If cdf=True cumulative distribution is plotted
    Passes any keyword arguments to matplotlib plot function
    '''
    x = x_range
    if cdf:
        y = ss.beta.cdf(x, alpha, beta)
    else:
        y = w*ss.beta.pdf(x, alpha, beta)
    if normalise:
        y /= np.sum(y)
    plt.plot(x, y,color=color, **kwargs)


x = np.linspace(0, 1, 500)
for d in md.mixt_comp:
    print d.alpha_0, d.beta_0
    plot_beta(x, d.alpha_0, d.beta_0)
    
    
    
mixt_comp = [ BetaBinomial(p=mu,M_0=2) for mu in means]


import emcee
import corner
from scipy.special import gammaln
from utils import get_mu_linear, exp_
import pandas as pd
import numpy as np
from scipy import logaddexp
from scipy.stats import pareto

files = !ls /Users/inti.pedroso/DATA/ASE/phaser_gene/
files
data = pd.concat([ pd.read_table( "".join(["/Users/inti.pedroso/DATA/ASE/phaser_gene/",f])) for f in files ])


data.head()
data2 = data[data["totalCount"] > 10]
data2.shape


this_data = data2.loc[data2.bam == "SRR5125126",["aCount","totalCount"]].values.astype(float)
this_data = this_data[this_data[:,0] > 10, :] 

def zs_to_alphas(zs):  
    """Project the hypercube coefficients onto the simplex"""
    fac = np.concatenate((1 - zs, np.array([1])))
    zsb = np.concatenate((np.array([1]), zs))
    fs = np.cumprod(zsb) * fac
    return fs

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

K = 15
means = get_mu_linear(K=K)
# When sampling CRP parameter
ndim, nwalkers = K + 1, 100
pos = np.vstack([ np.random.rand(ndim) for i in range(nwalkers)])
pos[:,:K] = pos[:,:K]*200 + 10
pos[:,K] = pos[:,K]*25 + 4

# When sampling CRP parameter
ndim, nwalkers = K, 100
pos = np.vstack([ np.random.rand(ndim) for i in range(nwalkers)])
pos[:,:K] = pos[:,:K]*200 + 10

sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=([means, this_data]), threads=1)
pos, prob, state = sampler.run_mcmc(pos, 200, progress=True)
    
print("Mean acceptance fraction: {0:.3f}".format(np.mean(sampler.acceptance_fraction)))

samples = sampler.chain[:, 100:, :].reshape((-1, ndim))
post_M  = samples.mean(0)[:K]
l_like = lnlike(samples.mean(0), this_data, means, return_pi=True)
post_pi = l_like['pi']

for i, (mu,m) in enumerate(zip(means,post_M)):
    print mu*m, (1.0-mu)*m
    plot_beta(x, mu*m, (1.0-mu)*m, w=2.5*post_pi[i], normalise=False)
plt.hist(this_data[:,0]/this_data[:,1], normed=True, bins=50, color='lightgrey')

_ = corner.corner(samples)


M = np.ones(K)*10
bnds = [(10, 500)]
bnds = bnds*len(M)
res = minimize(lnprob,x0=M, bounds=bnds, tol=1e-6)


