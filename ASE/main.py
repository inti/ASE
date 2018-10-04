#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 10:35:54 2018

@author: inti.pedroso
"""

import pandas as pd
import numpy as np
import emcee
import argparse

from distributions import lnprob, lnlike, get_mu_linear, get_observation_post
from stats import prob_1_diff_2


parser = argparse.ArgumentParser(description='Estimate Allelic Specific Expression probabilities')
parser.add_argument('--n_iter', type=int, default=10, help='Number of MCMC iterations')
parser.add_argument('--burnin', type=int, default=None, help='Number of MCMC Burnin iterations')
parser.add_argument('--thin', type=int, default=2, help='Thin every number of MCMC samples')
parser.add_argument('--n_walkers', type=int, default=30, help='Number of EMCEE walkers')

parser.add_argument('--input', type=str, nargs='+', default=None, help='Input files to consider')
parser.add_argument('--output', type=str, default=None, help='Input files to consider')
parser.add_argument('--min_allele_count', type=int, default=10, help='Min number of alternative allele count')
parser.add_argument('--min_total_count', type=int, default=10, help='Min total count')

parser.add_argument('--a_column', type=str, default="aCount", help='Name of column with counts for altertative allele')
parser.add_argument('--b_column', type=str, default="bCount", help='Name of column with counts for reference allele')
parser.add_argument('--sample_id_column', type=str, default="bCount", help='Name of column with Sample identifier')
parser.add_argument('--id', type=str, default="bCount", help='Name of column with row unique identifier')

parser.add_argument('--K', type=int, default=7, help='Number of mixture components')
parser.add_argument('--n_cores', type=int, default=1, help='Number of mixture components')
parser.add_argument('--not_compute_on_unique_counts', action='store_true', type=bool, default=False, help='SHould likelihood be computed over unique count pairs to speed up calculations? (Default: True)')


args = parser.parse_args()

print "Reading input data"
data = pd.concat([ pd.read_table( f) for f in args.input ])

print "Read [ ", data.shape[0]," ] data points"

data.loc[:,"tmp_total"] = data.loc[:,[args.a_column, args.b_column]].sum(1).values

count_tuple_frequency = None
if args.not_compute_on_unique_counts:
    grp = data.groupby([args.a_column, "tmp_total"])
    count_data = grp.size().reset_index(name="count_frq").values
else:
    count_data = data.loc[:,[args.a_column, "tmp_total"]].values.astype(float)
# filter by min counts
count_data = count_data[count_data[:,0] >= args.min_allele_count, :] 
count_data = count_data[count_data[:,:2].sum(1) >= args.min_total_count, :] 

if args.not_compute_on_unique_counts:
    count_tuple_frequency = count_data[:,2]
    count_data = count_data[:,:2]

print "   '-> After filteting for min counts there are [ ", count_data.shape[0]," ] data points"

# get fix mean values
means = get_mu_linear(K=args.K)

# prepapre MCMCM sampler
print "MCMC sampling"
ndim, args.n_walkers = args.K, 100
pos = np.vstack([ np.random.rand(ndim) for i in range(args.n_walkers)])
pos[:,:args.K] = pos[:,:args.K]*200 + args.min_allele_count

sampler = emcee.EnsembleSampler(args.n_walkers, 
                                ndim, 
                                lnprob, 
                                args=([means, count_data, count_tuple_frequency]), 
                                threads=args.n_cores)
pos, prob, state = sampler.run_mcmc(pos, args.n_iter, progress=True)
    
print("Mean acceptance fraction: {0:.3f}\n".format(np.mean(sampler.acceptance_fraction)))

samples = sampler.chain[:, args.burnin::args.thin, :].reshape((-1, args.K))
post_M  = samples.mean(0)[:args.K]
l_like = lnlike(samples.mean(0), count_data, means, return_pi=True)
post_pi = l_like['pi']

pars = np.array([means * post_M, (1.0 - means)*post_M]).T

df_count_data = pd.DataFrame(data.loc[:,[args.a_column, "tmp_total"]].values.astype(float))  #pd.DataFrame(count_data)
df_count_data_unique = df_count_data.drop_duplicates(keep="first")


post_counts = get_observation_post(np.vstack([ df_count_data_unique.loc[:,0].values ,  
                                              df_count_data_unique.loc[:,1].values - df_count_data_unique.loc[:,0].values ]).T, 
                                    pars, 
                                    ncores=args.n_cores, 
                                    chunk=args.n_cores*5)


df2 = pd.merge(df_count_data,
         pd.DataFrame(np.hstack([ df_count_data_unique.values, post_counts ] )),
         how="left",
         sort=False)

df2.columns = [args.a_column,"tmp_total","alpha_post","beta_post"]

null_pars = pars[(args.K-1)/2,:]

df2.loc[:,"pASE"] = df2.apply(lambda x: prob_1_diff_2(x['alpha_post'],x['beta_post'],null_pars[0],null_pars[1]),axis=1)

out = pd.merge(data,
         df2,
         how="left",
         sort=False,
         left_on=[args.a_column,"tmp_total"],
         right_on=[args.a_column,"tmp_total"]).drop_duplicates()




out.to_csv(args.output, sep="\t",index=False,header=True)