#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 16:54:06 2018

@author: inti.pedroso
"""
import numpy as np
import logging
logger = logging.getLogger()

from scipy.special import betaln as lbeta, gammaln
from scipy.optimize import minimize
import scipy.integrate as integrate
from scipy import logaddexp, stats as ss
from utils import exp_

def invert_order(x):
    return np.array(x)[::-1]

def prob_2_beats_1_binary(alpha_1, beta_1, alpha_2, beta_2, invert_to_seep_up=True, pseucount=1, integrate=False):
    alpha_1 += pseucount
    beta_1 += pseucount
    alpha_2 += pseucount
    beta_2 += pseucount        
    
    if invert_to_seep_up:
        if (( alpha_2 > alpha_1) and (alpha_1 > 1)) or ((alpha_2 < alpha_1) and (alpha_1 == 1)):
            alpha_1, alpha_2 = invert_order([alpha_1,alpha_2])
            beta_1,  beta_2  = invert_order([beta_1, beta_2])
    if integrate:
        prob, error= prob_2_beats_1_binary_integrate(alpha_1, beta_1, alpha_2, beta_2)
    else:
        total = list()
        for i in xrange(0,int(np.around(alpha_2-1))):
            total.append( lbeta(alpha_1+i,beta_1 + beta_2) - np.log(beta_2+i) - lbeta(1+i,beta_2) - lbeta(alpha_1,beta_1))
        prob =  exp_(logaddexp.reduce(total))
    return prob

def prob_2_beats_1_binary_integrate(alpha_1, beta_1, alpha_2, beta_2, invert_to_seep_up=True, pseucount=1):
    alpha_1 += pseucount
    beta_1 += pseucount
    alpha_2 += pseucount
    beta_2 += pseucount        
       
    beta_prod = lambda x,y: ss.beta.pdf(x,alpha_1, beta_1,)*ss.beta.pdf(y,alpha_2, beta_2)
    
    g = lambda x: x # lower limit for second integral
    h = lambda y: 1 # upper limit for second integral
    prob = integrate.dblquad(beta_prod, 0, 1, g, h)
    return prob

def prob_2_beats_1_counts(alpha_1, beta_1, alpha_2, beta_2, invert_to_seep_up=True, pseucount=1):

    alpha_1 += pseucount
    beta_1 += pseucount
    alpha_2 += pseucount
    beta_2 += pseucount
    
    if invert_to_seep_up:
        if ((alpha_2 < alpha_1) and (alpha_2 > 1)) or ((alpha_2 > alpha_1) and (alpha_1 == 1)):
            alpha_1, alpha_2 = invert_order([alpha_1,alpha_2])
            beta_1,  beta_2  = invert_order([beta_1, beta_2])

    total = list()
    for k in xrange(0,int(np.around(alpha_1-1))):
        total.append(k*np.log(beta_1) + alpha_2*np.log(beta_2) - (k+alpha_2)*np.log(beta_1 + beta_2) - np.log(k+alpha_2) - lbeta(k+1,alpha_2))
    prob = exp_(logaddexp.reduce(total))
    return prob


 
    
def prob_1_diff_2(alpha_1, beta_1, alpha_2, beta_2, p_type="binary",invert_to_seep_up=True, pseucount=1, integrate=False):
    
    if p_type == "counts":
        p = prob_2_beats_1_counts(alpha_1, beta_1, alpha_2, beta_2,invert_to_seep_up=invert_to_seep_up, pseucount=pseucount)
    elif p_type == "binary":
        p = prob_2_beats_1_binary(alpha_1, beta_1, alpha_2, beta_2,invert_to_seep_up=invert_to_seep_up, pseucount=pseucount,integrate=integrate)
    
    else:
        print "Only probability types of [ binary ] and [ counts ] are allowed"
    
    if not integrate:
        p = np.abs(p - (1-p))
    return np.abs(p - (1-p))

def prob_ASE_mixt_prior(alpha=None,beta=None,pars=None, null_pars=None, prior_w = None,invert_to_seep_up=True, pseucount=1, integrate=False):
    local_K = pars.shape[0]
    if null_pars is None:
        null_pars = pars[(local_K-1)/2,:]
    if prior_w is None:
        prior_w = 1.0 - np.array([ prob_1_diff_2(pars[i,0],pars[i,1],null_pars[0],null_pars[1],invert_to_seep_up=invert_to_seep_up, pseucount=pseucount, integrate=integrate) for i in xrange(local_K)])
    prior_w /= prior_w.sum()
    prob_ASE = np.array([ prob_1_diff_2(alpha,beta,pars[i,0],pars[i,1],invert_to_seep_up=invert_to_seep_up, pseucount=pseucount, integrate=integrate) for i in xrange(local_K)])
    mixt_prob_ASE = np.dot(prob_ASE, prior_w)
    return mixt_prob_ASE


def probASE1(alpha,beta,alpha_null,beta_null,invert_to_seep_up=True, pseucount=1, tuple_count=None, integrate=False):
    p_type="counts"
    if integrate:
        p_type="binary"
        
    prob = prob_1_diff_2(alpha,beta,alpha_null,beta_null,invert_to_seep_up=invert_to_seep_up, p_type=p_type,pseucount=pseucount,integrate=integrate)
    if tuple_count is not None:
        if (prob == np.inf) | (prob > 1.0):
            prob = prob_1_diff_2(tuple_count[0], 
                                 tuple_count[1],
                                 alpha_null,
                                 beta_null,
                                 invert_to_seep_up=invert_to_seep_up,
                                 pseucount=pseucount)
    return prob


def probASE2(alpha=None,beta=None,pars=None, null_pars=None, prior_w = None,invert_to_seep_up=True, pseucount=1, tuple_count=None, integrate=False):
    prob = prob_ASE_mixt_prior(alpha=alpha,beta=beta,pars=pars, null_pars=null_pars, prior_w = prior_w,invert_to_seep_up=invert_to_seep_up, pseucount=pseucount, integrate=integrate)
    if tuple_count is not None:
        if (prob == np.inf) | (prob > 1.0):
            prob = prob_ASE_mixt_prior(tuple_count[0],tuple_count[1],pars=pars, null_pars=null_pars, prior_w = prior_w,invert_to_seep_up=invert_to_seep_up, pseucount=pseucount)
    return prob


def probASE3(alpha,beta,pars,weights=None,p_null=0.5, return_odds=False, prior_odds = None):
    if weights is None:
        weights = np.ones((pars.shape[0],))
    
    if prior_odds is None:
        prior_odds = 1.0

    x_points = np.linspace(1e-8,1-1e-8,500)
    # density on null
    log_p_M_null = logaddexp.reduce([ get_normalized_pdf_at(p_null, p[0], p[1], x_points=x_points) for p in pars]  +  np.log(weights))
    # density on alternative
    log_p_M_alternative = get_normalized_pdf_at(p_null, alpha,beta, x_points=x_points)

    
    
    logBF_01 = log_p_M_null - log_p_M_alternative
    log_post_odds_01 = logBF_01 + np.log(prior_odds)    
    if return_odds:
        return log_post_odds_01, [log_p_M_null, log_p_M_alternative]
    log_prob = exp_(log_post_odds_01 - np.logaddexp.reduce([log_post_odds_01, 1.0])) 
    return log_prob

def get_normalized_pdf_at(p,alpha,beta, x_points=None, return_log=True):
    if x_points is None:
        x_points = np.linspace(1e-8,1-1e-8,500)
    log_p = ss.beta.logpdf(p, alpha,beta)
    # nornmalise marginal density
    log_p -= np.log(integrate.quad(lambda x: ss.beta.pdf(x,alpha,beta),0,1))[0]
    if return_log:
        return log_p
    else:
        return exp_(log_p)
    
    



def stick_breaking_eb(x,method='L-BFGS-B',bounds=[(1,None)], x0=None, gamma_hyperprior = None):
    """
    
    Hyper-prior for Stick Breaking parameter: array with two values to be used in sicpy.stats.gamma.pdf(x,gamma_hyperprior[0], scale=gamma_hyperprior[1]). See https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gamma.html
    """
    if x is None:
        return np.nan
    def beta_logpdf(beta):
        
        back =  -np.sum(np.log(1 - x)*(beta - 1.0) - (gammaln(1) + gammaln(beta) - gammaln(1 + beta)))
        logger.debug("data likelihood: [%s]", str(back))
        if gamma_hyperprior is not None:
            hyper_like = -ss.gamma.logpdf(beta,gamma_hyperprior[0], scale=gamma_hyperprior[1])
            back += hyper_like
            logger.debug("hyperprior likelihood: [%s]", str(hyper_like))
        return back
    
    if x0 is None:
        # obtain mom
        x0 = 1.0/np.mean(x) - 1.0
    res = minimize(beta_logpdf, x0, method=method, bounds=bounds)
    if res.success:
        logger.debug("Returning ML estimate: [%s]", str(res.x[0]))
        return res.x[0]
    else:
        logger.debug("Returning MOM estimate: [%s]", str(x0))
        return x0
