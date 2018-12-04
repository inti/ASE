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
import yaml

# progress bar
from tqdm import tqdm
tqdm.pandas(desc="")

from ASE.distributions import lnprob_full, get_mu_linear, get_observation_post, unfold_symmetric_parameters
from ASE.stats import prob_1_diff_2, probASE1, probASE2, probASE3




parser = argparse.ArgumentParser(description='Estimate Allelic Specific Expression probabilities')
parser.add_argument('--n_iter', type=int, default=10, help='Number of MCMC iterations')
parser.add_argument('--burnin', type=int, default=None, help='Number of MCMC Burnin iterations')
parser.add_argument('--thin', type=int, default=2, help='Thin every number of MCMC samples')
parser.add_argument('--n_walkers', type=int, default=30, help='Number of EMCEE walkers')
parser.add_argument('--integration_n_points', type=int, default=200, help='Number of points to use for integration for the conflation operations')
parser.add_argument('--prior_all_free', action='store_true', default=False, help='Model all parameters of mixture model. In particular referering to the total mass of the individual Beta-Binomial distributions')
parser.add_argument('--prior_symmetric', action='store_true', default=False, help='Model parameters of mixture model as symmetric')

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

n_parameters = int((args.K-1)*0.5 + 1)
pos = np.vstack([ np.random.rand(n_parameters) for i in range(args.n_walkers)])
pos[:,:n_parameters] = pos[:,:n_parameters]*200 + args.min_allele_count

pool = schwimmbad.choose_pool(mpi=False, processes=args.n_cores)

sampler = emcee.EnsembleSampler(args.n_walkers, 
                                n_parameters, 
                                lnprob_full, 
                                args=([means, count_data, count_tuple_frequency]), 
                                pool=pool)
pos, prob, state = sampler.run_mcmc(pos, args.n_iter, progress=True)
pool.close()

logger.info("EMCEE Mean acceptance fraction: [ %0.3f ]", np.mean(sampler.acceptance_fraction))

samples = sampler.chain[:, args.burnin::args.thin, :].reshape((-1, n_parameters))
post_M  = unfold_symmetric_parameters(np.percentile(samples,'50',axis=0)) #samples.mean(0)[:args.K]
l_like = lnprob_full(np.percentile(samples,'50',axis=0),means, count_data, count_tuple_frequency, return_all_results=True)
post_pi = l_like['pi']


pars = np.array([means * post_M, (1.0 - means)*post_M]).T

logger.info("Fitted model parameters")
logger.info("Component %5s  Prevalence Mean  Alpha_post Beta_post")

for i in xrange(args.K):
    logger.info("   '-> %i %0.3f %0.3f %0.3f %0.3f", i, post_pi[i], means[i],pars[i,0],pars[i,1])

logger.info("Writing components parameters to [ %s ]", args.output + '.mixture_parameters.yaml')
pars_dict = dict()
pars_dict['components'] = dict()
for i in xrange(pars.shape[0]):
    pars_dict['components'][i] =  { 'alpha_post': float(pars[i,0]), 'beta_post': float(pars[i,1])}

pars_dict['run_info'] = {'n_iter': args.n_iter, 
                         'n_walkers': args.n_walkers,
                         'thin': args.thin,
                         'burnin': args.burnin,
                         'K': args.K,
                         'integration_n_points': args.integration_n_points,
                         'components_means': [ float(i) for i in means ],
                         'mean_acceptance_fraction': float(np.mean(sampler.acceptance_fraction)),
                         'min_allele_count': args.min_allele_count,
                         'min_total_count': args.min_total_count }

file_model_pars = open(args.output + '.mixture_parameters.yaml','w') 
yaml.dump(pars_dict, 
          stream=file_model_pars,
          Dumper=yaml.Dumper,
          default_flow_style=False,
          encoding=None)
file_model_pars.close()

df_count_data = pd.DataFrame(data.loc[:,[args.a_column, "tmp_total"]].values.astype(float), columns=[args.a_column, "tmp_total"])  #pd.DataFrame(count_data)
if args.test_only:
    df_count_data = df_count_data.loc[:100,:]
    
df_count_data_unique = df_count_data.drop_duplicates(keep="first")

logger.debug("Count Unique data head \n%s\n", df_count_data_unique.head().to_string())


logger.info("Calculating posterior distribution for observed counts")

post_counts = get_observation_post(counts = np.vstack([ df_count_data_unique.values[:,0] ,  
                                                        df_count_data_unique.values[:,1] - df_count_data_unique.values[:,0] 
                                                       ]).T, 
                                    prior_pars= pars,
                                    weights = post_pi,
                                    x_n_points=args.integration_n_points,
                                    ncores=args.n_cores, 
                                    chunk=args.n_cores*5)

logger.debug("Posterior Count data head \n%s\n", pd.DataFrame(post_counts[:10,:]).head().to_string())

df_count_data_unique = pd.DataFrame( np.hstack([ df_count_data_unique.values, post_counts ] ),
                      columns=[args.a_column,"tmp_total","alpha_post","beta_post"]
                      )


logger.info("Calculating probability of ASE: method 1")
null_pars = pars[(args.K-1)/2,:]
logger.debug("   '-> Null paramerers for pASE calculation [ %s ] and [ %s ]", null_pars[0],null_pars[1])
df_count_data_unique.loc[:,"pASE_1"] = df_count_data_unique.progress_apply(lambda x: probASE1(alpha=x['alpha_post'], beta=x['beta_post'], alpha_null=null_pars[0], beta_null=null_pars[1], invert_to_seep_up=True, pseucount=1, tuple_count=[x[args.a_column],x["tmp_total"]-x[args.a_column]]),axis=1)


logger.info("Calculating probability of ASE: method 2")
        
# get weight informing of the prob that each component is null 
prior_w = 1.0 - np.array([ prob_1_diff_2(pars[i,0],pars[i,1],null_pars[0],null_pars[1]) for i in xrange(args.K)])

df_count_data_unique.loc[:,"pASE_2"] = df_count_data_unique.progress_apply(lambda x: probASE2(x['alpha_post'],x['beta_post'],pars, null_pars=null_pars, prior_w=prior_w,invert_to_seep_up=True, pseucount=1, tuple_count=[x[args.a_column],x["tmp_total"]-x[args.a_column]]),axis=1)


# prob ASE 3
logger.info("Calculating probability of ASE: method 3")
df_count_data_unique.loc[:,"pASE_3"] = df_count_data_unique.progress_apply(lambda x: probASE3(alpha=x['alpha_post'],beta=x['beta_post'],pars=pars,weights=post_pi,p_null=0.5),axis=1)


df_count_data_unique.loc[:,"log2_aFC_post"] = np.log2(df_count_data_unique.alpha_post.values/df_count_data_unique.beta_post.values)
logger.debug("pASE calculation data head \n%s\n", df_count_data_unique.head().to_string())

df2 = pd.merge(df_count_data,
         df_count_data_unique,
         how="left", 
         on = [args.a_column,"tmp_total"],
         sort=False)

logger.debug("Merge data with posterior counts head \n%s\n", df2.head().to_string())



logger.info("Merging results with original data")

    
out = pd.merge(data,
         df2.drop_duplicates(),
         how="right",
         sort=False,
         left_on=[args.a_column,"tmp_total"],
         right_on=[args.a_column,"tmp_total"],
         validate="many_to_one")

logger.debug("Output head \n%s\n", out.head().to_string())


logger.info("Writting output file to [ %s ]", args.output + '.ase.txt')

out.to_csv(args.output + 'ase.txt', sep="\t",index=False,header=True)

logger.info("Done")


