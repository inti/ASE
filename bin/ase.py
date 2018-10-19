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
import logging
import schwimmbad

from ASE.distributions import lnprob, lnlike, get_mu_linear, get_observation_post
from ASE.stats import prob_1_diff_2



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
parser.add_argument('--not_compute_on_unique_counts', action='store_false', default=True, help='Should likelihood be computed over unique count pairs to speed up calculations? (Default: True)')

parser.add_argument('--test_only', action='store_true', default=False, help='Run on at most 100 data points and 10 iterations')
parser.add_argument('--debug', action='store_true', default=False, help='Pring debugging messages')

args = parser.parse_args()


logger = logging.getLogger()
handler = logging.StreamHandler()
formatter = logging.Formatter(
        '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logging_level = logging.INFO
if args.debug:
    logging_level = logging.DEBUG
logger.setLevel(logging_level)

#logging.basicConfig(filename='app.log', 
#                    filemode='w', 
#                    format='%(name)s - %(levelname)s - %(message)s',
#                    level=logging.WARNING)

logger.warning("Log file written to [ %s]", ''.join([args.output,".log"]))


logger.info("Reading input data")
data = []
for f in args.input:
    logger.debug("Reading file [ %s ]", f)
    data.append( pd.read_table(f) )
data = pd.concat(data)

logger.debug("Read data head \n%s\n", data.head().to_string())

logger.info("Read [ %s ]  data points",data.shape[0])
logger.info("Reading A and B allele counts from columns [ %s ] and [ %s ]", args.a_column, args.b_column)
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

logger.debug("Count data head \n%s\n", pd.DataFrame(count_data[:10,:]).head().to_string())


if args.test_only:
    count_data = count_data[:100,:]
    args.n_iter = 10
    args.thin = 2
    args.burnin = 2

if args.not_compute_on_unique_counts:
    count_tuple_frequency = count_data[:,2]
    count_data = count_data[:,:2]


logger.info("   '-> After filteting for min counts there are [ %s ] data points", count_data.shape[0])

# get fix mean values
means = get_mu_linear(K=args.K)

# prepapre MCMCM sampler
logger.info("MCMC sampling")
logger.info("Mixture model with [ %i ] components", args.K)
logger.info("   '-> N CPUs : %i", args.n_cores)
logger.info("   '-> N iterations : %i", args.n_iter)
logger.info("   '-> N burnin : %i", args.burnin)
logger.info("   '-> N thin : %i", args.thin)
logger.info("EMCEE parameters: nwalkers : [ %i ]", args.n_walkers)

pos = np.vstack([ np.random.rand(args.K) for i in range(args.n_walkers)])
pos[:,:args.K] = pos[:,:args.K]*200 + args.min_allele_count

pool = schwimmbad.choose_pool(mpi=False, processes=args.n_cores)

sampler = emcee.EnsembleSampler(args.n_walkers, 
                                args.K, 
                                lnprob, 
                                args=([means, count_data, count_tuple_frequency]), 
                                pool=pool)
pos, prob, state = sampler.run_mcmc(pos, args.n_iter, progress=True)
pool.close()

logger.info("EMCEE Mean acceptance fraction: [ %0.3f ]", np.mean(sampler.acceptance_fraction))

samples = sampler.chain[:, args.burnin::args.thin, :].reshape((-1, args.K))
post_M  = samples.mean(0)[:args.K]
l_like = lnlike(samples.mean(0), count_data, means, return_pi=True)
post_pi = l_like['pi']


pars = np.array([means * post_M, (1.0 - means)*post_M]).T

logger.info("Fitted model parameters")
logger.info("Component %5s  Prevalence Mean  Alpha_post Beta_post")

for i in xrange(args.K):
    logger.info("   '-> %i %0.3f %0.3f %0.3f %0.3f", i, post_pi[i], means[i],pars[i,0],pars[i,1])


    
df_count_data = pd.DataFrame(data.loc[:,[args.a_column, "tmp_total"]].values.astype(float))  #pd.DataFrame(count_data)
if args.test_only:
    df_count_data = df_count_data.loc[:100,:]
    
df_count_data_unique = df_count_data.drop_duplicates(keep="first")

logger.debug("Count Unique data head \n%s\n", df_count_data_unique.head().to_string())


logger.info("Calculating posterior distribution for observed counts")
post_counts = get_observation_post(np.vstack([ df_count_data_unique.loc[:,0].values ,  
                                              df_count_data_unique.loc[:,1].values - df_count_data_unique.loc[:,0].values ]).T, 
                                    pars, 
                                    ncores=args.n_cores, 
                                    chunk=args.n_cores*5)

logger.debug("Posterior Count data head \n%s\n", pd.DataFrame(post_counts[:10,:]).head().to_string())

df2 = pd.merge(df_count_data,
         pd.DataFrame(np.hstack([ df_count_data_unique.values, post_counts ] )),
         how="left", 
         sort=False)

df2.columns = [args.a_column,"tmp_total","alpha_post","beta_post"]

logger.debug("Merge data with posterior counts head \n%s\n", df2.head().to_string())


null_pars = pars[(args.K-1)/2,:]

logger.debug("Null paramerers for pASE calculation [ %s ] and [ %s ]", null_pars[0],null_pars[1])
logger.info("Calculating probability of ASE")
df2.loc[:,"pASE"] = df2.apply(lambda x: prob_1_diff_2(x['alpha_post'],x['beta_post'],null_pars[0],null_pars[1]),axis=1)

logger.debug("pASE calculation data head \n%s\n", df2.head().to_string())

logger.info("Merging results with original data")

#for c in ['alpha_post', 'beta_post', 'pASE']:
#    data.loc[:,c] = np.nan
    
#def add_post_values(key,grp, values):
#    data.loc[grp.groups[key],['alpha_post', 'beta_post', 'pASE']] = values
    
#grp = data.groupby([args.a_column,"tmp_total"])
#out = df2.apply(lambda x: add_post_values((x[args.a_column],x["tmp_total"]),
#                                          grp,
#                                          x[['alpha_post', 'beta_post', 'pASE']]),
#                        axis=1)
    
out = pd.merge(data,
         df2.drop_duplicates(),
         how="right",
         sort=False,
         left_on=[args.a_column,"tmp_total"],
         right_on=[args.a_column,"tmp_total"],
         validate="many_to_one")

logger.debug("Output head \n%s\n", out.head().to_string())


logger.info("Writting output file to [ %s ]", args.output)

out.to_csv(args.output, sep="\t",index=False,header=True)

logger.info("Done")


