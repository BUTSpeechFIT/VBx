#!/usr/bin/env python

# Copyright 2021 Lukas Burget, Mireia Diez (burget@fit.vutbr.cz, mireia@fit.vutbr.cz)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Revision History
#   L. Burget   20/1/2021 1:00AM - original version derived from the more
#                                  complex VB_diarization.py avaiable at
# https://github.com/BUTSpeechFIT/VBx/blob/e39af548bb41143a7136d08310765746192e34da/VBx/VB_diarization.py
#

import numpy as np
from scipy.special import logsumexp


def VBx(X, Phi, loopProb=0.9, Fa=1.0, Fb=1.0, pi=10, gamma=None, maxIters=10,
        epsilon=1e-4, alphaQInit=1.0, ref=None, plot=False,
        return_model=False, alpha=None, invL=None):
    """
    Inputs:
    X           - T x D array, where columns are D dimensional feature vectors
                  (e.g. x-vectors) for T frames
    Phi         - D array with across-class covariance matrix diagonal.
                  The model assumes zero mean, diagonal across-class and
                  identity within-class covariance matrix.
    loopProb    - Probability of not switching speakers between frames
    Fa          - Scale sufficient statiscits
    Fb          - Speaker regularization coefficient Fb controls the final number of speakers
    pi          - If integer value, it sets the maximum number of speakers
                  that can be found in the utterance.
                  If vector, it is the initialization for speaker priors (see Outputs: pi)
    gamma       - An initialization for the matrix of responsibilities (see Outputs: gamma)
    maxIters    - The maximum number of VB iterations
    epsilon     - Stop iterating, if the obj. fun. improvement is less than epsilon
    alphaQInit  - Dirichlet concentraion parameter for initializing gamma
    ref         - T dim. integer vector with per frame reference speaker IDs (0:maxSpeakers)
    plot        - If set to True, plot per-frame marginal speaker posteriors 'gamma'
    return_model- Return also speaker model parameter
    alpha, invL - If provided, these are speaker model parameters used in the first iteration

    Outputs:
    gamma       - S x T matrix of responsibilities (marginal posteriors)
                  attributing each frame to one of S possible speakers
                  (S is defined by input parameter pi)
    pi          - S dimensional column vector of ML learned speaker priors.
                  This allows us to estimate the number of speaker in the
                  utterance as the probabilities of the redundant speaker
                  converge to zero.
    Li          - Values of auxiliary function (and DER and frame cross-entropy
                  between gamma and reference, if 'ref' is provided) over iterations.
    alpha, invL - Speaker model parameters returned only if return_model=True

    Reference:
      Landini F., Profant J., Diez M., Burget L.: Bayesian HMM clustering of
      x-vector sequences (VBx) in speaker diarization: theory, implementation
      and analysis on standard tasks
    """
    """
    The comments in the code refers to the equations from the paper above. Also
    the names of variables try to be consistent with the symbols in the paper.
    """

    D = X.shape[1]  # feature (e.g. x-vector) dimensionality

    if type(pi) is int:
        pi = np.ones(pi)/pi

    if gamma is None:
        # initialize gamma from flat Dirichlet prior with
        # concentration parameter alphaQInit
        gamma = np.random.gamma(alphaQInit, size=(X.shape[0], len(pi)))
        gamma = gamma / gamma.sum(1, keepdims=True)

    assert(gamma.shape[1] == len(pi) and gamma.shape[0] == X.shape[0])

    G = -0.5*(np.sum(X**2, axis=1, keepdims=True) + D*np.log(2*np.pi))  # per-frame constant term in (23)
    V = np.sqrt(Phi)  # between (5) and (6)
    rho = X * V  # (18)
    Li = []
    for ii in range(maxIters):
        # Do not start with estimating speaker models if those are provided
        # in the argument
        if ii > 0 or alpha is None or invL is None:
            invL = 1.0 / (1 + Fa/Fb * gamma.sum(axis=0, keepdims=True).T*Phi)  # (17) for all speakers
            alpha = Fa/Fb * invL * gamma.T.dot(rho)  # (16) for all speakers
        log_p_ = Fa * (rho.dot(alpha.T) - 0.5 * (invL+alpha**2).dot(Phi) + G)  # (23) for all speakers
        tr = np.eye(len(pi)) * loopProb + (1-loopProb) * pi  # (1) transition probability matrix
        gamma, log_pX_, logA, logB = forward_backward(log_p_, tr, pi)  # (19) gamma, (20) logA, (21) logB, (22) log_pX_
        ELBO = log_pX_ + Fb * 0.5 * np.sum(np.log(invL) - invL - alpha**2 + 1)  # (25)
        pi = gamma[0] + (1-loopProb)*pi * np.sum(np.exp(logsumexp(
            logA[:-1], axis=1, keepdims=True) + log_p_[1:] + logB[1:] - log_pX_
        ), axis=0)  # (24)
        pi = pi / pi.sum()
        Li.append([ELBO])

        # if reference is provided, report DER, cross-entropy and plot figures
        if ref is not None:
            Li[-1] += [DER(gamma, ref), DER(gamma, ref, xentropy=True)]

            if plot:
                import matplotlib.pyplot
                if ii == 0:
                    matplotlib.pyplot.clf()
                matplotlib.pyplot.subplot(maxIters, 1, ii+1)
                matplotlib.pyplot.plot(gamma, lw=2)
                matplotlib.pyplot.imshow(np.atleast_2d(ref),
                                         interpolation='none', aspect='auto',
                                         cmap=matplotlib.pyplot.cm.Pastel1,
                                         extent=(0, len(ref), -0.05, 1.05))

        if ii > 0 and ELBO - Li[-2][0] < epsilon:
            if ELBO - Li[-2][0] < 0:
                print('WARNING: Value of auxiliary function has decreased!')
            break
    return (gamma, pi, Li) + ((alpha, invL) if return_model else ())


# Calculates Diarization Error Rate (DER) or per-frame cross-entropy between
# reference (vector of per-frame zero based integer speaker IDs) and gamma
# (per-frame speaker posteriors). If expected=False, gamma is converted into
# hard labels before calculating DER. If expected=TRUE, posteriors in gamma
# are used to calculated "expected" DER.
def DER(q, ref, expected=True, xentropy=False):
    from scipy.sparse import coo_matrix
    from scipy.optimize import linear_sum_assignment
    if not expected:  # replce probabilities in q by zeros and ones
        q = coo_matrix((np.ones(len(q)), (range(len(q)), q.argmax(1)))).toarray()

    ref_mx = coo_matrix((np.ones(len(ref)), (range(len(ref)), ref)))
    err_mx = ref_mx.T.dot(-np.log(q+np.nextafter(0, 1)) if xentropy else -q)
    min_cost = err_mx[linear_sum_assignment(err_mx)].sum()
    return min_cost/float(len(ref)) if xentropy else (len(ref) + min_cost)/float(len(ref))


def forward_backward(lls, tr, ip):
    """
    Inputs:
        lls - matrix of per-frame log HMM state output probabilities
        tr  - transition probability matrix
        ip  - vector of initial state probabilities (i.e. starting in the state)
    Outputs:
        pi  - matrix of per-frame state occupation posteriors
        tll - total (forward) log-likelihood
        lfw - log forward probabilities
        lfw - log backward probabilities
    """
    eps = 1e-8
    ltr = np.log(tr + eps)
    lfw = np.empty_like(lls)
    lbw = np.empty_like(lls)
    lfw[:] = -np.inf
    lbw[:] = -np.inf
    lfw[0] = lls[0] + np.log(ip + eps)
    lbw[-1] = 0.0

    for ii in range(1, len(lls)):
        lfw[ii] = lls[ii] + logsumexp(lfw[ii-1] + ltr.T, axis=1)

    for ii in reversed(range(len(lls)-1)):
        lbw[ii] = logsumexp(ltr + lls[ii+1] + lbw[ii+1], axis=1)

    tll = logsumexp(lfw[-1], axis=0)
    pi = np.exp(lfw + lbw - tll)
    return pi, tll, lfw, lbw
