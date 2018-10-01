#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 10:35:54 2018

@author: inti.pedroso
"""

from distributions import lnprob
from utils import get_mu_linear, exp_
import pandas as pd
import numpy as np


import emcee
import corner
from scipy.special import gammaln
from utils import get_mu_linear, exp_
import pandas as pd
import numpy as np

files = !ls /Users/inti.pedroso/DATA/ASE/phaser_gene/
files
data = pd.concat([ pd.read_table( "".join(["/Users/inti.pedroso/DATA/ASE/phaser_gene/",f])) for f in files ])


data.head()
data2 = data[data["totalCount"] > 10]
data2.shape


this_data = data2.loc[data2.bam == "SRR5125126",["aCount","totalCount"]].values.astype(float)
this_data = this_data[this_data[:,0] > 10, :] 


K = 7
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


