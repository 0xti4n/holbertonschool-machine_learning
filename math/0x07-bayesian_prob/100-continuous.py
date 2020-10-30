#!/usr/bin/env python3
"""Continuous Posterior Bayesian prob"""
from scipy import special


def posterior(x, n, p1, p2):
    """calculates the posterior probability that the probability
        of developing severe side effects falls within a
        specific range given the data:

    -> x is the number of patients that develop severe side effects

    -> n is the total number of patients observed

    -> p1 is the lower bound on the range

    -> p2 is the upper bound on the range

    -> You can assume the prior beliefs of p follow a uniform distribution

    -> Returns: the posterior probability that p is within
        the range [p1, p2] given x and n
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError('n must be a positive integer')

    if not isinstance(x, int) or x < 0:
        msg = 'x must be an integer that is greater than or equal to 0'
        raise ValueError(msg)

    if x > n:
        raise ValueError('x cannot be greater than n')

    if not isinstance(p1, float) or p1 < 0 or p1 > 1:
        raise ValueError('p1 must be a float in the range [0, 1]')

    if not isinstance(p2, float) or p2 < 0 or p2 > 1:
        raise ValueError('p2 must be a float in the range [0, 1]')

    if p2 <= p1:
        raise ValueError('p2 must be greater than p1')

    """
        The binomial distribution is the PMF of k successes
        given n independent events each with a probability
        p of success. Mathematically,
        when α = k + 1 and β = n − k + 1,
        the beta distribution and the binomial
        distribution are related by a factor of n + 1
    https://en.wikipedia.org/wiki/Binomial_distribution
    """

    a = x + 1
    beta = n - x + 1

    b_cdf = special.btdtr(a, beta, p1)
    b_cdf2 = special.btdtr(a, beta, p2)

    b_pdf = b_cdf2 - b_cdf

    return b_pdf
