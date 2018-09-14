import os

import numpy as np
from astropy.io import fits
from scipy.ndimage import convolve

def fits_write(a, path, file_name):
    if file_name in os.listdir(path):
        os.remove(os.path.join(path, file_name))
    fits.PrimaryHDU(a).writeto(os.path.join(path, file_name))

def main(size=[1000, 1000]):
    if 'tests' not in os.listdir():
        os.mkdir('tests')

    noise_path = lambda s: '../../data/noise/{}.fits'.format(s)
    tinytim_path = lambda s: '../../data/tinytim/{}.fits'.format(s)
    for b in 'hjvz':
        tiny_tim = fits.getdata(tinytim_path(b))
        noise = fits.getdata(noise_path(b))
        test = np.random.choice(noise, size=size)
        fits_write(convolve(test, tiny_tim), './tests', '{}.fits'.format(b))

if __name__=='__main__':
    main()
