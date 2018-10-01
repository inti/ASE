#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 10:30:34 2018

@author: inti.pedroso
"""

from scipy.optimize import minimize
from scipy import logaddexp
import scipy.stats as ss
import numpy as np
        
def beta_mom(mean=None,variance=None):
    """
        Returns the parameters of a beta distributions using the method of moments
    """
    
    common_factor = mean*((1-mean)/variance-1)
    return [mean*common_factor,   (1-mean)*common_factor]

def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    variance = np.average((values-average)**2, weights=weights)  # Fast and numerically precise
    return (average, np.sqrt(variance))

def beta_pars_from_cdf(cdf_vals = None, x_points=None,weights=None):
    """
        Infers the parameters of a beta distributions starting from the CDF by minimising the first and second moments of the distribution
    """
    m,s2 = weighted_avg_and_std(x_points,weights=cdf_vals)
    
    def beta_pdf_delta(beta_par,local_m=m,local_s2=s2, x_points = x_points, weights=None):
        proposed_cdf = get_beta_cdf([beta_par], x_points=x_points, weights=None) 
        proposed_m, proposed_s2 =  weighted_avg_and_std(x_points, weights= proposed_cdf)
        delta =  (local_m - proposed_m)**2 + (local_s2 - proposed_s2)**2
        return(delta)
    
    x0 = beta_mom(m, s2)
    res = minimize(beta_pdf_delta, x0, method='nelder-mead', options={'xtol': 1e-8,'disp': False})
    back = res.x.astype(list).astype(float)
    return( [back[0],back[1]] )



def get_beta_cdf(pars,x_range=[0,1], x_n_points=500, x_points=None, weights=None):
    """
        For a set of beta distribitions parametrised as [[alpha_1,beta_1],[alpha_2,beta_2]] obtained the conflated distribution.
        See function beta_conflation for more details 
    """
    if weights is None:
        weights = np.ones(len(pars))
    if len(weights) != len(pars):
        raise ValueError('Number of weights must be equal to number of parameters')
    
    weights /= np.max(weights)
    
    if x_points is None:
        x_points = np.linspace(x_range[0], x_range[1], x_n_points)
    it = iter(pars)
    initializer = next(it)

    w_it = iter(weights)
    w_init = next(w_it)
    log_accum_value = ss.beta.logpdf(x_points, initializer[0], initializer[1])
    w_sum = w_init

    for next_pars in it:
        w_value = next(w_it)
        w_prop = w_sum/(w_sum + w_value)
        # log (first^w_prop)*(second/(1-w_prop)) = log(first) - log(w_prop) + log(second) - log((1-w_prop))
        log_accum_value = w_prop*log_accum_value + (1.0-w_prop)*ss.beta.logpdf(x_points, next_pars[0], next_pars[1])
        w_sum += w_value
    return np.exp(log_accum_value - logaddexp.reduce(log_accum_value))


def beta_conflation(pars=None, x_range=[1e-8,1.0-1e-8], x_n_points=500, initializer=None, weights=None):
    """
        Performs conflation on beta distributions following Hill (2011, https://arxiv.org/pdf/0808.1808.pdf) and 
        Hill and Miller (2011, http://dx.doi.org/10.1063/1.3593373)
        
        pars = parameters of the beta distributions list of list where each list provides the pair alpha beta: e.g. [[alpha_1,beta_1],[alpha_2,beta_2]]
        x_range = value range to consider the conflation list of to values [min, max]
        x_n_points= number of points to evalue on the x_range
        initializer = initial set of paramaters to consider 
        weights = weights to combine the differet distributions. Internally will be normalised by dividing by max value as per Hill and Miller (2011)
        
        Return a list of two values [alpha, beta] corrsponding to the parameters of the combined distribution
        
    """
    x_points = np.linspace(x_range[0], x_range[1], x_n_points)
    
    it = iter(pars)
    if initializer is None:
        try:
            initializer = next(it)
        except StopIteration:
            raise TypeError('Sequence with no initial value')
            
    if weights is None:
        weights = np.ones(len(pars))
    if len(weights) != len(pars):
        raise ValueError('Number of weights must be equal to number of parameters')
    
    weights /= np.max(weights)
    
    accum_value = initializer
    
    w_it = iter(weights)
    w_init = next(w_it)
    
    w_sum = w_init
    for next_pars in it:
        w_value = next(w_it)
        cdf_vals = get_beta_cdf([accum_value,next_pars], x_points=x_points, weights=[w_sum, w_value])
        accum_value = beta_pars_from_cdf(cdf_vals, x_points=x_points,weights=weights)
        w_sum += w_value

    return accum_value