import sys
sys.path.append('../../model')

import os

import numpy as np
from astropy.io import fits

import make_test
from inference import Classifier

def main():
    # make tests if they don't exist
    if 'tests' not in os.listdir():
        make_test.main()

    out_dir = './output'
    if 'output' not in os.listdir():
        os.mkdir(out_dir)

    h = './tests/h.fits'
    j = './tests/j.fits'
    v = './tests/v.fits'
    z = './tests/z.fits'

    Classifier.classify_files(h=h, j=j, v=v, z=z, out_dir=out_dir, batch_size=1000)

    morphs = [
        'spheroid',
        'disk',
        'irregular',
        'point_source',
        'background',
    ]
    
    n = fits.getdata('./output/n.fits')
    
    for morph in morphs:
        a = fits.getdata('./output/{}.fits'.format(morph))[-1, :, :]
        
        normed_to_n = np.divide(a, n, out=np.zeros_like(a), where=n!=0)
       
        fits.PrimaryHDU(data=normed_to_n).writeto('./output/top_{}.fits'.format(morph),
                                                  overwrite=True)

if __name__=='__main__':
    main()


