#!/usr/bin/env python

# Copyright Brno University of Technology
# Licensed under the Apache License, Version 2.0 (the "License")

import numpy as np
import scipy.linalg as spl
import numexpr as ne
import h5py


def gmm_eval(data, GMM, return_accums=0):
    """ llh = GMM_EVAL(d,GMM) returns vector of log-likelihoods evaluated for each
    frame of dimXn_samples data matrix using GMM object. GMM object must be
    initialized with GMM_EVAL_PREP function.

    [llh N F] = GMM_EVAL(d,GMM,1) also returns accumulators with zero, first order statistics.

    [llh N F S] = GMM_EVAL(d,GMM,2) also returns accumulators with second order statistics.
    For full covariance model second order statistics, only the vectorized upper
    triangular parts are stored in columns of 2D matrix (similarly to GMM.invCovs).
    """
    # quadratic expansion of data
    data_sqr = data[:, GMM['utr']] * data[:, GMM['utc']]  # quadratic expansion of the data
    # computation of log-likelihoods for each frame and all Gaussian components
    gamma = -0.5 * data_sqr.dot(GMM['invCovs'].T) + data.dot(GMM['invCovMeans'].T) + GMM['gconsts']
    llh = logsumexp(gamma, axis=1)

    if return_accums == 0:
        return llh

    gamma = np.exp(gamma.T - llh)
    N = gamma.sum(axis=1)
    F = gamma.dot(data)
    if return_accums == 1:
        return llh, N, F

    S = gamma.dot(data_sqr)
    return llh, N, F, S


def gmm_eval_prep(weights, means, covs):
    n_mix, dim = means.shape
    GMM = dict()
    is_full_cov = covs.shape[1] != dim
    GMM['utr'], GMM['utc'] = uppertri_indices(dim, not is_full_cov)

    if is_full_cov:
        GMM['gconsts'] = np.zeros(n_mix)
        GMM['invCovs'] = np.zeros_like(covs)
        GMM['invCovMeans']=np.zeros_like(means)
        for ii in range(n_mix):
            uppertri1d_to_sym(covs[ii], GMM['utr'], GMM['utc'])
            invC, logdetC = inv_posdef_and_logdet(uppertri1d_to_sym(covs[ii], GMM['utr'], GMM['utc']))

            #log of Gauss. dist. normalizer + log weight + mu' invCovs mu
            invCovMean = invC.dot(means[ii])
            GMM['gconsts'][ii] = np.log(weights[ii]) - 0.5 * (logdetC + means[ii].dot(invCovMean) + dim * np.log(2.0*np.pi))
            GMM['invCovMeans'][ii] = invCovMean

            #Iverse covariance matrices are stored in columns of 2D matrix as vectorized upper triangular parts ...
            GMM['invCovs'][ii] = uppertri1d_from_sym(invC, GMM['utr'], GMM['utc']);
        # ... with elements above the diagonal multiplied by 2
        GMM['invCovs'][:,dim:] *= 2.0
    else: #for diagonal
        GMM['invCovs']  = 1 / covs;
        GMM['gconsts']  = np.log(weights) - 0.5 * (np.sum(np.log(covs) + means**2 * GMM['invCovs'], axis=1) + dim * np.log(2.0*np.pi))
        GMM['invCovMeans'] = GMM['invCovs'] * means

    # for weight = 0, prepare GMM for uninitialized model with single Gaussian
    if len(weights) == 1 and weights[0] == 0:
        GMM['invCovs']     = np.zeros_like(GMM['invCovs'])
        GMM['invCovMeans'] = np.zeros_like(GMM['invCovMeans'])
        GMM['gconsts']     = np.ones(1)
    return GMM


def gmm_llhs(data, GMM):
    """ llh = GMM_EVAL(d,GMM) returns vector of log-likelihoods evaluated for each
    frame of dimXn_samples data matrix using GMM object. GMM object must be
    initialized with GMM_EVAL_PREP function.

    [llh N F] = GMM_EVAL(d,GMM,1) also returns accumulators with zero, first order statistics.

    [llh N F S] = GMM_EVAL(d,GMM,2) also returns accumulators with second order statistics.
    For full covariance model second order statistics, only the vectorized upper
    triangular parts are stored in columns of 2D matrix (similarly to GMM.invCovs).
    """
    # quadratic expansion of data
    data_sqr = data[:, GMM['utr']] * data[:, GMM['utc']]  # quadratic expansion of the data
    # computation of log-likelihoods for each frame and all Gaussian components
    gamma = -0.5 * data_sqr.dot(GMM['invCovs'].T) + data.dot(GMM['invCovMeans'].T) + GMM['gconsts']

    return gamma


def gmm_update(N,F,S):
    """ weights means covs = gmm_update(N,F,S) return GMM parameters, which are
    updated from accumulators
    """
    dim = F.shape[1]
    is_diag_cov = S.shape[1] == dim
    utr, utc = uppertri_indices(dim, is_diag_cov)
    sumN    = N.sum()
    weights = N / sumN
    means   = F / N[:,np.newaxis]
    covs    = S / N[:,np.newaxis] - means[:,utr] * means[:,utc]
    return weights, means, covs


def inv_posdef_and_logdet(A):
    L = np.linalg.cholesky(A)
    logdet = 2*np.sum(np.log(np.diagonal(L)))
    invA = spl.solve(A, np.identity(len(A), A.dtype), sym_pos=True)
    return invA, logdet


def logsumexp(x, axis=0):
    xmax = x.max(axis)
    x = xmax + np.log(np.sum(np.exp(x - np.expand_dims(xmax, axis)), axis))
    not_finite = ~np.isfinite(xmax)
    x[not_finite] = xmax[not_finite]
    return x


#def logsumexp_numexpr(x, axis=0):
#    xmax = x.max(axis)
#    xmax_e = np.expand_dims(xmax, axis)
#    t = ne.evaluate("sum(exp(x - xmax_e),axis=%d)" % axis)
#    x =  ne.evaluate("xmax + log(t)")
#    not_finite = ~np.isfinite(xmax)
#    x[not_finite] = xmax_e[not_finite]
#    return x


def uppertri_indices(dim, isdiag=False):
    """ [utr utc]=uppertri_indices(D, isdiag) returns row and column indices
    into upper triangular part of DxD matrices. Indices go in zigzag fashion
    starting by diagonal. For convenient encoding of diagonal matrices, 1:D
    ranges are returned for both outputs utr and utc when ISDIAG is true.
    """
    if isdiag:
        utr = np.arange(dim)
        utc = np.arange(dim)
    else:
        utr = np.hstack([np.arange(ii)     for ii in range(dim,0,-1)])
        utc = np.hstack([np.arange(ii,dim) for ii in range(dim)])
    return utr, utc


if(__name__ == "__main__"):
    pass
