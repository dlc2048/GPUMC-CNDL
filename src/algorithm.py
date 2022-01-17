import numpy as np
import matplotlib.pyplot as plt
from scipy.special import legendre, eval_legendre
from scipy import interpolate

def legendreToPolynomial(coeff):
    """
        convert legendre coefficient array to polynomial
    """
    polynomial = np.zeros(len(coeff), dtype=np.float64)
    for order, val in enumerate(coeff):
        polynomial[-1-order:] += val * legendre(order)
    return polynomial

def legendreToEquibin(coeff, nbin, mu_min=-1.0, mu_max=1.0):
    """
        convert legendre angle distribution to
        equiprobable angle bin distribution
        from mu_min to mu_max
    """
    if mu_min >= mu_max:
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
    """
        get logarithm mean
    """
    return (b - a) / (np.log(b) - np.log(a))


class interp1d:
    def __init__(self, x, y, int):
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
            raise ValueError("illegal int value")
    
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


def getInterpFtnCumulArea(xx, yy, x):
    if x<xx[0] or x>xx[-1]:
        raise ValueError("x must be in xx range")
    target_index = np.argmax(x <= xx)
    y = interp1d(xx, yy, 2).get(x)
    xx_new = np.append(xx[:target_index], x)
    yy_new = np.append(yy[:target_index], y)
    area = (xx_new[1:] - xx_new[:-1])*(yy_new[1:] + yy_new[:-1])/2
    return sum(area)

def getInterpFtnCumulValue(xx, yy, area):
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


if __name__ == "__main__":
    """
        test script for equiprobability angle conversion
    """
    legendre_coeff = np.array([1.00000000e+00, -9.54075725e-01,  8.69809366e-01,
                        -7.54805076e-01,  6.19137782e-01, -4.77040938e-01,
                         3.40556182e-01, -2.19013776e-01,  1.20502852e-01], dtype=np.float64)
    modifier = (np.arange(0, len(legendre_coeff), 1) * 2 + 1) / 2
    legendre_coeff *= modifier

    nbin = 32
    angle_bin = legendreToEquibin(legendre_coeff, nbin)
    y = 1 / (angle_bin[1:] - angle_bin[:-1]) / nbin
    plt.step(angle_bin[:-1], y)
    plt.yscale("log")
    plt.show()