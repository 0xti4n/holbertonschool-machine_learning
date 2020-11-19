#!/usr/bin/env python3
"""Bayesian Optimization"""
import numpy as np
from scipy.stats import norm
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization():
    """performs Bayesian optimization
    on a noiseless 1D Gaussian process"""

    def __init__(self, f, X_init, Y_init, bounds, ac_samples,
                 l=1, sigma_f=1, xsi=0.01, minimize=True):
        """ Constructor

        -> f is the black-box function to be optimized

        -> X_init is a numpy.ndarray of shape (t, 1) representing
            the inputs already sampled with the black-box function

        -> Y_init is a numpy.ndarray of shape (t, 1) representing
            the outputs of the black-box function for each input in X_init

            * t is the number of initial samples

        -> bounds is a tuple of (min, max) representing the
            bounds of the space in which to look for the optimal point

        -> ac_samples is the number of samples that should be
            analyzed during acquisition

        -> l is the length parameter for the kernel

        -> sigma_f is the standard deviation given to the
            output of the black-box function

        -> xsi is the exploration-exploitation factor for acquisition

        -> minimize is a bool determining whether optimization
            should be performed for minimization (True) or maximization (False)
        """
        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)
        self.X_s = np.linspace(bounds[0], bounds[1], ac_samples).reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """calculates the next best sample location
            Expected Improvement acquisition function

        Returns: X_next, EI
            * X_next is a numpy.ndarray of shape (1,)
                representing the next best sample point

            * EI is a numpy.ndarray of shape (ac_samples,)
                containing the expected improvement of each potential sample
        """
        mu_s, _ = self.gp.predict(self.gp.X)
        mu, sigma = self.gp.predict(self.X_s)

        sigma = sigma.reshape(-1, 1)
        mu_s_opt = np.max(mu_s)

        if self.minimize is True:
            mu_s_opt = np.min(self.gp.Y)
            imp = mu_s_opt - mu - self.xsi
        else:
            mu_s_opt = np.amax(self.gp.Y)
            imp = mu - mu_s_opt - self.xsi

        with np.errstate(divide='warn'):
            imp = imp.reshape(-1, 1)
            Z = imp / sigma
            EI = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            EI[sigma == 0.0] = 0.0

        X_next = self.X_s[np.argmax(EI)]
        return X_next, np.squeeze(EI.T)

    def optimize(self, iterations=100):
        """optimizes the black-box function:

        -> iterations is the maximum number
            of iterations to perform

        -> Returns: X_opt, Y_opt
            * X_opt is a numpy.ndarray of shape (1,)
                representing the optimal point
            * Y_opt is a numpy.ndarray of shape (1,)
                representing the optimal function value
        """
        for i in range(iterations):
            X_opt, ei = self.acquisition()

            if X_opt in self.gp.X:
                self.gp.X = self.gp.X[:-1]
                break

            Y_opt = self.f(X_opt)
            self.gp.update(X_opt, Y_opt)

        if self.minimize:
            idx = np.argmin(self.gp.Y)
        else:
            idx = np.argmax(self.gp.Y)

        X_opt = self.gp.X[idx]
        Y_opt = self.gp.Y[idx]

        return X_opt, Y_opt
