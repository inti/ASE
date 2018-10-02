#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 10:35:54 2018

@author: inti.pedroso
"""

import pandas as pd
import numpy as np
import emcee
import corner
from distributions import lnprob, lnlike, get_mu_linear, exp_, get_observation_post
from scipy.special import gammaln

import argparse

parser = argparse.ArgumentParser(description='Estimate Allelic Specific Expression probabilities')
parser.add_argument('--n_iter', type=int, default=10, help='Number of MCMC iterations')
parser.add_argument('--burnin', type=int, default=None, help='Number of MCMC Burnin iterations')
parser.add_argument('--thin', type=int, default=2, help='Thin every number of MCMC samples')

parser.add_argument('--input', type=str, nargs='+', default=None, help='Input files to consider')
parser.add_argument('--output', type=str, nargs='+', default=None, help='Input files to consider')
parser.add_argument('--min_allele_count', type=int, default=10, help='Min number of alternative allele count')
parser.add_argument('--min_total_count', type=int, default=10, help='Min total count')

parser.add_argument('--a_column', type=str, default="aCount", help='Name of column with counts for altertative allele')
parser.add_argument('--b_column', type=str, default="bCount", help='Name of column with counts for reference allele')
parser.add_argument('--sample_id_column', type=str, default="bCount", help='Name of column with Sample identifier')
parser.add_argument('--id', type=str, default="bCount", help='Name of column with row unique identifier')

parser.add_argument('--K', type=int, default=7, help='Number of mixture components')
parser.add_argument('--n_cores', type=int, default=1, help='Number of mixture components')


args = parser.parse_args()

print "Reading input data"
data = pd.concat([ pd.read_table( f) for f in args.input ])

print "Read [ ", data.shape[0]," ] data points"

count_data = data.loc[:,[args.a_column, args.b_column]].values.astype(float)
# filter by min counts
count_data = count_data[count_data[:,0] >= args.min_allele_count, :] 
count_data = count_data[count_data.sum(1) >= args.min_total_count, :] 
print "   '-> After filteting for min counts there are [ ", count_data.shape[0]," ] data points"


K = 7
# get fix mean values
means = get_mu_linear(K=args.K)

# prepapre MCMCM sampler
print "MCMC sampling"
ndim, nwalkers = K, 100
pos = np.vstack([ np.random.rand(ndim) for i in range(nwalkers)])
pos[:,:K] = pos[:,:K]*200 + args.min_allele_count

sampler = emcee.EnsembleSampler(nwalkers, 
                                ndim, 
                                lnprob, 
                                args=([means, count_data]), 
                                threads=args.n_cores)
pos, prob, state = sampler.run_mcmc(pos, args.n_iter, progress=True)
    
print("Mean acceptance fraction: {0:.3f}\n".format(np.mean(sampler.acceptance_fraction)))

samples = sampler.chain[:, args.burnin::args.thin, :].reshape((-1, K))
post_M  = samples.mean(0)[:K]
l_like = lnlike(samples.mean(0), count_data, means, return_pi=True)
post_pi = l_like['pi']

#for i, (mu,m) in enumerate(zip(means,post_M)):
#    print mu*m, (1.0-mu)*m
#    plot_beta(mu*m, (1.0-mu)*m, w=2.5*post_pi[i], normalise=False)
#plt.hist(this_data[:,0]/this_data[:,1], density=True, bins=50, color='lightgrey')

#_ = corner.corner(samples)


pars = np.array([means * post_M, (1.0 - means)*post_M]).T

post_counts = get_observation_post(count_data, pars, ncores=args.n_cores)

