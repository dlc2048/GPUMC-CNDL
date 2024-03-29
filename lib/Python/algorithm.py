#!/usr/bin/env python3
"""Provides interpolation algorithm that follows ENDF interpolation law,
and equi-probable angular bin conversion algorithm from legendre polynomial

this code is part of the GPU-accelerated Monte Carlo project
"""

import numpy as np
from scipy.special import legendre
from scipy import interpolate

from lib.Python import constants

__copyright__ = "Copyright 2021, GPUMC Project"
__license__ = "MIT"
__author__ = "Chang-Min Lee"
__email__ = "dlc2048@postech.ac.kr"
__status__ = "Production"


def legendreToPolynomial(coeff):
    """convert legendre coefficient array to polynomial"""
    polynomial = np.zeros(len(coeff), dtype=np.float64)
    for order, val in enumerate(coeff):
        polynomial[-1-order:] += val * legendre(order)
    return polynomial

def legendreToEquibin(coeff, nbin, mu_min=-1.0, mu_max=1.0):
    """convert legendre angle distribution to
    equiprobable angle bin distribution
    from mu_min to mu_max

    coeff: legendre coefficient series
    nbin: the number of equiprobable angle bin
    mu_min: lower bound of directional cosine
    mu_max: upper bound of directional cosine
    """
    if mu_min > mu_max:
        raise ValueError("mu_min must be smaller than mu_max")
        
    poly = legendreToPolynomial(coeff)

    # find roots, only real number
    roots = np.roots(poly)
    roots = np.real(roots[np.isreal(roots)])
    roots = roots[(mu_min < roots) * (roots < mu_max)]
    roots = np.sort(roots)
    roots = np.unique(roots)
    if mu_min not in roots:
        roots = np.append(mu_min, roots)
    if mu_max not in roots:
        roots = np.append(roots, mu_max)

    # get integral
    polyint = np.poly1d(np.polyint(poly))
    # get area between each neighboring roots
    area_cumul = polyint(roots)
    area = area_cumul[1:] - area_cumul[:-1]
    area_total = np.sum(area[area > 0])
    area_seg = area_total / nbin
    last_area = 0

    # find equiprob angle bin
    angle_bin = np.empty(nbin+1, dtype=np.float64)
    angle_bin[0] = mu_min
    angle_bin[-1] = mu_max
    n = 1
    for i in range(len(area)):
        if area[i] <= 0:
            continue
        root_lower = roots[i]
        root_upper = roots[i+1]
        int_lower = polyint(root_lower)
        while True: # get answer
            polyint_t = np.copy(polyint)
            polyint_t[-1] += -int_lower + last_area - n * area_seg
            roots_int = np.roots(polyint_t)
            roots_int = np.real(roots_int[np.isreal(roots_int)])
            roots_int = roots_int[(root_lower <= roots_int) * (roots_int <= root_upper)]
            if len(roots_int) > 0:
                angle_bin[n] = np.min(roots_int)
                n += 1
            else:
                break
        last_area += area[i]

    return angle_bin, area_total

def logMean(a, b):
    """get logarithm mean"""
    return (b - a) / (np.log(b) - np.log(a))


class interp1d:
    def __init__(self, x, y, int):
        """one-dimensional interpolation, follows ENDF interpolation law

        x: domain of the function. should be the numpy array
        y: value of the function. should be the numpy array
        int: ENDF interpolation law.
        """
        self._int = int
        if self._int == 2: # linear-linear
            self._f = interpolate.interp1d(x, y)
        elif self._int == 3: # linear-log
            self._f = interpolate.interp1d(x, np.log(y))
        elif self._int == 4: # log-linear
            self._f = interpolate.interp1d(np.log(x), y)
        elif self._int == 5: # log-log
            self._f = interpolate.interp1d(np.log(x), np.log_y)
        else:
            raise ValueError("illegal interpolation law")
    
    def get(self, x):
        if self._int == 2:
            y = self._f(x)
        elif self._int == 3:
            log_y = self._f(x)
            y = np.exp(log_y)
        elif self._int == 4:
            y = self._f(np.log(x))
        elif self._int == 5:
            log_y = self._f(np.log(x))
            y = np.exp(log_y)
        return y


class interp2d:
    def __init__(self, x, y, z, int):
        """two-dimensional interpolation, follows ENDF interpolation law

        x: domain of the function. should be the numpy array
        y: domain of the function. should be the numpy array
        z: value of the function. should be the numpy array
           shape must be (nx, ny)
        int: ENDF interpolation law.
        """
        self._int = int
        if self._int == 2: # linear-linear
            self._f = interpolate.interp2d(x, y, z)
        elif self._int == 3: # linear-log
            # 0 point handling
            z_copy = np.copy(z)
            z_copy[z_copy < constants.LOG_MIN] = constants.LOG_MIN
            self._f = interpolate.interp2d(x, y, np.log(z_copy))
        elif self._int == 4: # log-linear
            x_copy = np.copy(x)
            y_copy = np.copy(y)
            x_copy[x_copy < constants.LOG_MIN] = constants.LOG_MIN
            y_copy[y_copy < constants.LOG_MIN] = constants.LOG_MIN
            self._f = interpolate.interp2d(np.log(x_copy), np.log(y_copy), z)
        elif self._int == 5: # log-log
            z_copy = np.copy(z)
            z_copy[z_copy < constants.LOG_MIN] = constants.LOG_MIN
            x_copy = np.copy(x)
            y_copy = np.copy(y)
            x_copy[x_copy < constants.LOG_MIN] = constants.LOG_MIN
            y_copy[y_copy < constants.LOG_MIN] = constants.LOG_MIN
            self._f = interpolate.interp2d(np.log(x_copy), np.log(y_copy), np.log(z_copy))
        else:
            raise ValueError("illegal interpolation law")

    def get(self, x, y):
        if self._int == 2:
            z = self._f(x, y)
        elif self._int == 3:
            log_z = self._f(x, y)
            z = np.exp(log_z)
        elif self._int == 4:
            z = self._f(np.log(x), np.log(y))
        elif self._int == 5:
            log_z = self._f(np.log(x), np.log(y))
            z = np.exp(log_z)
        return z


def getInterpFtnCumulArea(xx, yy, x):
    """get the area of numerical function (xx, yy) from xx[0] to x

    xx: domain of the function. should be the numpy array
    yy: value of the function. should be the numpy array
    x: upper bound of domain.

    note: this function uses rectangular rule. it should be improved
    """
    if x<xx[0] or x>xx[-1]:
        raise ValueError("x must be in xx range")
    target_index = np.argmax(x <= xx)
    y = interp1d(xx, yy, 2).get(x)
    xx_new = np.append(xx[:target_index], x)
    yy_new = np.append(yy[:target_index], y)
    area = (xx_new[1:] - xx_new[:-1])*(yy_new[1:] + yy_new[:-1])/2
    return sum(area)

def getInterpFtnCumulValue(xx, yy, area):
    """get the upper bound x when the integrated area of
    numerical function (xx, yy) from xx[0] to x is the "area"

    xx: domain of the function. should be the numpy array
    yy: value of the function. should be the numpy array
    area: area of integral

    note: this function uses rectangular rule. it should be improved
    """
    area_seg = (xx[1:] - xx[:-1])*(yy[1:] + yy[:-1])/2
    area_cumul = np.cumsum(area_seg)
    target_index = np.argmax(area < area_cumul)
    a = area - np.sum(area_seg[:target_index])
    s = (yy[target_index+1]-yy[target_index])/(xx[target_index+1]-xx[target_index])
    b = s * xx[target_index] - yy[target_index]
    c = s * xx[target_index]**2 - 2*yy[target_index]*xx[target_index] - 2*a
    if s != 0:
        x = (b+np.sqrt(b**2-s*c))/s
    else:
        x = a/yy[target_index] + xx[target_index]
    return x

class AliasTable:
    def __init__(self, domain, prob):
        """convert discrete probability density function to
        alias table

        domain: domain of the probability density function
        prob: value of the probability density function
        """
        self._domain = domain
        self._alias_table = np.empty(domain.shape, dtype=np.int32)
        self._prob_table = np.copy(prob)
        prob_tag = np.ones(prob.shape, dtype=np.bool)

        mean = np.sum(prob) / len(prob)
        # set alias table
        for i in range(len(prob) - 1):
            lower = np.where((self._prob_table < mean) * prob_tag)[0]
            upper = np.where((self._prob_table > mean) * prob_tag)[0]

            if len(lower) == 0 or len(upper) == 0:
                continue

            target_low = lower[0]
            target_up = upper[0]

            aux = mean - self._prob_table[target_low]
            self._prob_table[target_up] -= aux
            self._prob_table[target_low] /= mean
            self._alias_table[target_low] = target_up

            prob_tag[target_low] = False

        self._prob_table[prob_tag] = 10

    def sampling(self):
        rand = np.random.random()
        aj = rand * len(self._domain)
        j = int(aj)
        aj -= j
        if aj > self._prob_table[j]:
            j = self._alias_table[j]
        return j

    def getProbTable(self):
        return self._prob_table

    def getAliasTable(self):
        return self._alias_table

def probFromAlias(alias_table, alias_index):
    """convert alias table to the probability density function"""
    prob = np.zeros(alias_table.shape)
    mean = 1 / len(alias_table)
    for i in range(len(alias_table)):
        if alias_table[i] >= 1:
            prob[i] += mean
        else:
            target = alias_index[i]
            prob[target] += mean * (1.e0 - alias_table[i])
            prob[i] += mean * alias_table[i]
    
    return prob
