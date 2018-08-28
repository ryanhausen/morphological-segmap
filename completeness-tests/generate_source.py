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
        return _gen_source(size, cy, cx, f, f'n-{n}')

def _simple_gen_source(size, cy, cx, f):
    ys, xs = np.meshgrid(np.arange(size[0]), np.arange(size[1]))
    rs = np.sqrt((xs-cx)**2 + (ys-cy)**2)
    return f(rs)

def _gen_source(size, cy, cx, f, name):
    if f'cache_vals_{name}.json' in os.listdir():
        with open(f'cache_vals_{name}.json', 'r') as f:
            cache_vals = json.load(f)
    else:
        cache_vals = {}

    src = np.zeros(size)

    ys, xs = np.meshgrid(np.arange(size[0]), np.arange(size[1]))
    ys = ys - cy
    xs = xs - cx

    for i in range(size[0]):
        for j in range(size[1]):
            y = ys[i,j]
            x = xs[i,j]

            key = f'{abs(y)},{abs(x)}'
            if key not in cache_vals:
                val = dblquad(f, x-0.5, x+0.5, lambda a: y-0.5, lambda a: y+0.5)[0]
                cache_vals[key] = val
                cache_vals[f'{x},{y}'] = val


            src[i,j] = cache_vals[key]

    with open(f'cache_vals_{name}.json', 'w') as f:
        json.dump(cache_vals, f)

    return src

def main():
    size = [100, 100]
    re = 5
    ie = 1

    cy, cx = size[0]//2, size[1]//2
    ys, xs = np.meshgrid(np.arange(size[0]), np.arange(size[1]))
    ys = ys - cy
    xs = xs - cx

    rs = np.sqrt(xs**2 + ys**2)

    tt = fits.getdata('./tinytim/v.fits')

    exp = exponential(size, cy, cx, ie, re)
    dv = deVaucouleurs(size, cy, cx, ie, re)

    tt_exp = convolve(exp, tt)
    tt_dv = convolve(dv, tt)

    for img, name in zip([exp, dv, tt_exp, tt_dv], ['Exponential', 'de Vaucouleurs', 'Exp with TinyTim', 'de Vaucouleurs with Tiny Tim']):
        f, (a1, a2) = plt.subplots(nrows=2)
        f.suptitle(name)
        a1.imshow(np.log10(img), cmap='gray', origin='lower')

        sort_rs = np.argsort(rs.flatten())
        sorted_fs = img.flatten()[sort_rs]
        sorted_rs = np.array(sorted(rs.flatten()))

        a2.semilogy(sorted_rs, sorted_fs, 'b-')
        a2.set_xlabel('R (pixels)')
        a2.set_ylabel('Flux', color='b')
        a2.tick_params('y', colors='b')

        ratio_fs = sorted_fs.cumsum()/sorted_fs.sum()
        a3 = a2.twinx()
        a3.plot(sorted_rs, ratio_fs, 'r-')
        a3.set_ylabel('%Total Flux', color='r')
        a3.tick_params('y', colors='r')

        a3.hlines(0.5, sorted_rs.min(), sorted_rs.max(), color='r', alpha=0.5)
        a3.vlines(5.0, 0, 1, color='r', alpha=0.5)



        print(name, img[rs<=re].sum()/img.sum())

    plt.show()




if __name__=='__main__':
    main()
