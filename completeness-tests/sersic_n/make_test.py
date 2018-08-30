# include the parent dir to access generate_sources.py
import sys
sys.path.append('..')

import generate_source

import os

import numpy as np
from astropy.io import fits
from scipy.ndimage import convolve

def fits_write(a, path, file_name):
    if file_name in os.listdir(path):
        os.remove(os.path.join(path, file_name))
    fits.PrimaryHDU(a).writeto(os.path.join(path, file_name))


def get_rms(size, re, num_samples, noise):
    ys, xs = np.meshgrid(np.arange(size), np.arange(size))

    aperature_rs = np.sqrt((ys-size//2)**2 + (xs-size//2)**2)
    aperature = aperature_rs < re

    y_buffer = np.sum(aperature, axis=0).max() / 2
    x_buffer = np.sum(aperature, axis=1).max() / 2
    y_shift  = int(size//2 - y_buffer)
    x_shift  = int(size//2 - x_buffer)

    rms_vals = []
    for _ in range(num_samples):
        s0 = np.random.randint(y_shift) * 1 if np.random.randint(2) else -1
        s1 = np.random.randint(x_shift) * 1 if np.random.randint(2) else -1

        shifted_aperature = np.roll(aperature, (s0,s1), axis=(0,1))

        rms_vals.append(noise[shifted_aperature].sum())

    return np.sqrt(np.mean(np.square(rms_vals)))


def main(save_noise_separate=False, num_samples=1000):
    bands = 'hjvz'
    noise_path = lambda s: '../../data/noise/{}.fits'.format(s)
    noise = {b:fits.getdata(noise_path(b)) for b in bands}

    tinytim_path = lambda s: '../../data/tinytim/{}.fits'.format(s)
    tinytim = {b:fits.getdata(tinytim_path(b)) for b in bands}

    src_funcs = [
        ('sersic_n', generate_source.sersic_n),
    ]
    for name, src_func in src_funcs:
        for re in [3, 5, 7, 9]:
            print('Working on {} for re {}'.format(name, re))
            # make the noise that we'll add the sources to big enough so that we
            # can fit 3 sources across and 3 down with 6*re space between
            # each of the sources. Each source will need 6*re on either side
            # so in total 3 * (2 * 6re) pixels in both directions
            dim = 3 * (2 * 6 * re)
            all_noise = {}
            for b in bands:
                noise_img = np.random.choice(noise[b], size=[dim,dim])
                noise_img = noise_img + abs(noise_img.min()) + 1e-6
                all_noise[b] = noise_img

            if save_noise_separate:
                if 'noise' not in os.listdir():
                    os.mkdir('./noise')

                f_name = '{}-re-{}.fits'
                for b in bands:
                    fits_write(all_noise[b], './noise', f_name.format(b, re))

            center_ys = [6*re, 18*re, 30*re]
            center_xs = [6*re, 18*re, 30*re]

            ys, xs = np.meshgrid(np.arange(dim), np.arange(dim))

            centers = []
            rs = []

            for y in center_ys:
                for x in center_xs:
                    centers.append([y,x])
                    rs.append(np.sqrt((ys-y)**2 + (xs-x)**2))

            rms = {}
            for b in bands:
                rms[b] = get_rms(dim, re, num_samples, all_noise[b])

            sn_ratio = 10
            sersic_n_vals = [1, 2, 3, 4, 5, 6, 7, 8, 9]

            for i, n in enumerate(sersic_n_vals):
                raw_src = src_func([dim, dim],
                                   centers[i][0],
                                   centers[i][1],
                                   1,
                                   re,
                                   n,
                                   simple=False)
                source = {}
                for band in rms:
                    src_adj = raw_src * (sn_ratio * rms[band] / raw_src[rs[i]<re].sum())
                    src_adj = convolve(src_adj, tinytim[band])
                    source[band] = src_adj

                for b in bands:
                    all_noise[b] = all_noise[b] + source[b]

            img = np.array([all_noise[b] for b in 'hjvz'])

            if 'tests' not in os.listdir():
                os.mkdir('tests')

            fits_write(img, './tests', '{}-re-{}.fits'.format(name, re))

            for f in os.listdir('.'):
                if 'cache_vals' in f:
                    os.remove(f)

if __name__=='__main__':
    main()
