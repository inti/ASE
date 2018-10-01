#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 10:22:37 2018

@author: inti.pedroso
"""

from scipy.optimize import minimize
from scipy.stats import pareto
from scipy.special import gammaln
from scipy import logaddexp

import numpy as np
from tqdm import tnrange

from conflation import beta_conflation
from utils import exp_

class BetaBinomial(object):
    def __init__(self,alpha_0=None,beta_0=None,M_0=None,var_0=None,alpha=None,beta=None,p=None):
        assert p is not None
        self.name = 'BetaDistribution'
        self.p = p
        if (alpha_0 is None) and (beta_0 is None):
            if M_0:
                self.M_0 = M_0
                self.alpha_0 = self.p * self.M_0
                self.beta_0 = self.M_0 - self.alpha_0
            if var_0:
                self.alpha_0, self.beta_0 = self._beta_mom(self.p,self.var_0)
                self.M0 = self.alpha_0 + self.beta_0
        else:
            self.alpha_0 = alpha_0
            self.beta_0 = beta_0
        
        if alpha is None:
            self.alpha = self.alpha_0
            
        if beta is None:
            self.beta = self.beta_0
            
    def __str__(self):
        back = "BetaBinomial distribution: p [ %g ], alpha [ %g ], beta [ %g ]" % (self.p, self.alpha_0, self.beta_0)
        return back
    
    def _beta_mom(self,x_hat=None,s2=None):
        """Return parameters of Beta distribution taking as input the first two moments: mean and variance"""
        one_minus_mean = 1 - x_hat
        alpha = x_hat*( x_hat*one_minus_mean/s2 - 1.0)
        beta = alpha*one_minus_mean/x_hat
        return [alpha, beta]
    
    def _w_average_variance(self,x=None,w=None):
        if w is None:
            w = np.ones_like(x)
        x = x[w>0]
        w = w[w>0]
        mean = np.ma.average(x, axis=0, weights=w)
        xm = x-mean
        sigma2 = 1./(w.sum()-1) * np.multiply(xm,w).T.dot(xm);
        return [mean, sigma2]
    
    def _get_bb_moments_v2(self,k_vals=None,n_vals=None,weights=None):
        if weights is None:
            weights = np.ones_like(k_vals)
        idx = (n_vals>0) *(k_vals>0) * (weights > 0)
        assert np.sum(idx) > 1
        
        k = k_vals[idx]
        n = n_vals[idx]
        w = weights[idx]
        N = len(k)
        
        m1, m2 = self._w_average_variance(x=k/n,w=w)
        mu = np.sum(k*w)/np.sum(n*w)

        M = (mu*(1-mu) - m2)/(m2 - (mu*(1-mu)/N)*np.sum(1/n))
        s_2 = (1.0/N)*np.sum(w * (mu*(1-mu)/k)*(1+ (n-1)/(M+1)) )
        return [mu,s_2]

    def _fit(self, data= None, weights = None):
        """Update the alpha and beta parameters on the basis of some observations and some weights (optional)"""
        assert data is not None
        
        if weights is None:
            weights = np.ones(data.shape[0], dtype=float)
            
        m,s2 = self._get_bb_moments_v2(data[:,0],data[:,1],weights= weights )
        self.alpha, self.beta = self._beta_mom(m,s2) 
    
    def update(self, data= None, weights = None, x_n_points=100, keep_mean = False):
        """Update prior parameters alpha_0 and beta_0 by combining the data with the prior. 
        Update is done by conflation the two distribution &(Beta(self.alpha_0,self.beta_0),Beta(self.alpha,self.beta))"""
        self._fit(data= data, weights = weights)
        self.alpha_0,self.beta_0 = beta_conflation(pars=[[self.alpha_0,self.beta_0],[self.alpha,self.beta]], x_n_points=x_n_points)
        if keep_mean:
            M_0 = self.alpha_0 + self.beta_0
            self.alpha_0 = self.p * M_0
            self.beta_0 = M_0 - self.alpha_0
    
    def update_eb(self, init_M = 20, data=None, weights = None, pareto_par = 1.5 , min_val=10):
        # objevtive function to optimize
        def _fit_bb_ml_eb(pars, p, loca_data, w, pareto_par , min_val):
            M = pars[0]
            a,b = p*M, (1.0-p)*M
            fx = self._log_beta_binomial_density
            return np.dot(w, fx(loca_data[:,0], loca_data[:,1], a, b)) + pareto.logpdf(x=M, b=pareto_par, scale=min_val)

        if weights is None:
            weights = np.ones(data.shape[0])
        weights /= np.max(weights)
        
        bnds = [(min_val, None)]
        res = minimize(_fit_bb_ml_eb, 
                       x0=[init_M], 
                       args=(self.p, data,weights, pareto_par,min_val), 
                       bounds=bnds, 
                       tol=1e-6)
        self.alpha_0 = self.p * res.x[0]
        self.beta_0 = res.x[0] - self.alpha_0
        print self.p, self.alpha_0, self.beta_0, res.success
        
    def _log_beta_binomial_density(self,k,n,alpha,beta):
        uno = gammaln(n+1) - (gammaln(k+1) + gammaln(n-k+1))
        dos = gammaln(k+alpha) + gammaln(n-k+beta) - gammaln(n+alpha+beta)
        tres = gammaln(alpha + beta) - (gammaln(alpha) + gammaln(beta))
        return uno + dos + tres
    

    def logpdf(self,data=None):
            return [ self._log_beta_binomial_density(data[i,0],data[i,1],self.alpha,self.beta)  for i in xrange(data.shape[0])] 
    
    
class Mixt(object):
    """
    Define a Mixture distribution
    component_dist = list of individual distributions that form the mixture
    pi = mixture proportions 
    """
    def __init__(self, component_dist=[], pi = None, data= None, weights = None):
        self.K = len(component_dist)

        assert self.K > 1
        self.data = data
        self.weights = weights
        
        self.mixt_comp = component_dist
        if pi is None:
            pi = np.ones(self.K)/float(self.K)
        self.pi = pi
        assert self.K == self.pi.shape[0]
        
        self.log_z = None
        
        
    def __str__(self):
        back = """Mixture distribution: N Components [ %i ]""" % (len(self.mixt_comp))
        return back
    
    
    def _single_fit(self, fit_by_eb = True, null_iter=0):        
        if self.weights is None:
            self.weights = np.ones(self.data.shape[0])
        
        # calculate membership vectors
        self._update_mix_membership()
        
        # fit null distribution
        for i,dist in enumerate(self.mixt_comp):
            if dist.p == 0.5 and (null_iter > 0):
                for iter_ in xrange(null_iter):
                    w = exp_(self.log_z[:,i])
                    w = w/np.max(w)
                    if fit_by_eb:
                        dist.update_eb(data=self.data, weights = w)
                    else:
                        dist.update(data=self.data,
                                    weights = w,
                                    x_n_points=100,
                                    keep_mean=True)
                    self._update_mix_membership()
                    
        # update individual distributions
        for i,dist in enumerate(self.mixt_comp):
            w = exp_(self.log_z[:,i])
            print np.min(w),np.max(w)
            w = w/np.max(w)
            if fit_by_eb:
                dist.update_eb(data=self.data, weights = w)                
            else:
                dist.update(data=self.data,
                            weights = w,
                            x_n_points=100, 
                            keep_mean=True)

        self._update_mix_proportions()
       
    def _update_mix_membership(self):
        ## get the log density for each data point at each distribution
        logpdf = np.array([ dist.logpdf(data=self.data) for dist in self.mixt_comp]).T
        ## normalize the log densities to sum to 1 for each data point
        self.log_z = (logpdf.T - logaddexp.reduce(logpdf,axis=1)).T
        
    def _update_mix_proportions(self):
        # update mixture proportions
        pi = logaddexp.reduce(self.log_z,axis=0)
        self.pi = exp_(pi - logaddexp.reduce(pi))
        
    def fit(self, n_iter=1, delta = 1e-8, print_delta=False, fit_by_eb = True, null_iter=10):
        self.converged = False
        
        for i in tnrange(null_iter, desc="Fit null distribution"):
            current_params = self._extract_par_vector()
            self._single_fit(fit_by_eb=fit_by_eb, null_iter=null_iter)
            new_params = self._extract_par_vector()
            current_delta = np.sum(np.abs(current_params - new_params)/current_params)/current_params.shape[0]
            if current_delta <= delta:
                break
                
        for i in tnrange(n_iter, desc="Mixture fit"):
            current_params = self._extract_par_vector()
            self._single_fit(fit_by_eb=fit_by_eb, null_iter=0)
            new_params = self._extract_par_vector()
            current_delta = np.sum(np.abs(current_params - new_params)/current_params)/current_params.shape[0]
            if print_delta:
                print i, current_delta
            if current_delta <= delta:
                self.converged = True
                print "converged after [ %s ] iterations" % (i+1)
                break
        
        
    def _extract_par_vector(self):
        return np.hstack(
                        [self.pi, 
                         np.hstack(
                             [ [ dist.alpha_0, dist.beta_0] for dist in self.mixt_comp]) ]
                        )
    