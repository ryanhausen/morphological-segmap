from collections import namedtuple
import json
import os

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.integrate import quad, dblquad
from scipy.ndimage import convolve


def sersic(Ie, Re, R, n):
    bm = 2.0*n - 0.324
    return Ie * np.exp(-bm * ((R/Re)**(1/n) - 1))


def sersic_cart(Ie, Re, x, y, n):
    return sersic(Ie, Re, np.sqrt(x**2 + y**2), n)

def exponential(size, cy, cx, Ie, Re, simple=False):
    if simple:
        return _simple_gen_source(size, cy, cx, lambda r: sersic(Ie, Re, r, 1))
    else:
        f = lambda x, y: sersic_cart(Ie, Re, x, y, 1)
        return _gen_source(size, cy, cx, f, 'exponential')

def deVaucouleurs(size, cy, cx, Ie, Re, simple=False):
    if simple:
        return _simple_gen_source(size, cy, cx, lambda r: sersic(Ie, Re, r, 4))
    else:
        f = lambda x, y: sersic_cart(Ie, Re, x, y, 4)
        return _gen_source(size, cy, cx, f, 'deVaucouleurs')

def sersic_n(size, cy, cx, Ie, Re, n, simple=False):
    if simple:
        return _simple_gen_source(size, cy, cx, lambda r: sersic(Ie, Re, r, n))
    else:
        f = lambda x, y: sersic_cart(Ie, Re, x, y, n)
        return _gen_source(size, cy, cx, f, 'n-{}'.format(n))

def _simple_gen_source(size, cy, cx, f):
    ys, xs = np.meshgrid(np.arange(size[0]), np.arange(size[1]))
    rs = np.sqrt((xs-cx)**2 + (ys-cy)**2)
    return f(rs)

def _gen_source(size, cy, cx, src_func, name):
    if 'cache_vals_{}.json'.format(name) in os.listdir():
        with open('cache_vals_{}.json'.format(name), 'r') as f:
            cache_vals = json.load(f)
    else:
        cache_vals = {}

    src = np.zeros(size)

    ys, xs = np.meshgrid(np.arange(size[0]), np.arange(size[1]))
    ys = ys - cy
    xs = xs - cx

    for i in range(size[0]):
        for j in range(size[1]):
            y = abs(ys[i,j])
            x = abs(xs[i,j])

            key = '{},{}'.format(abs(y), abs(x))
            if key not in cache_vals:
                val = dblquad(src_func, x-0.5, x+0.5, lambda a: y-0.5, lambda a: y+0.5)[0]
                cache_vals[key] = val
                cache_vals['{},{}'.format(x,y)] = val


            src[i,j] = cache_vals[key]

    with open('cache_vals_{}.json'.format(name), 'w') as f:
        json.dump(cache_vals, f)

    return src
