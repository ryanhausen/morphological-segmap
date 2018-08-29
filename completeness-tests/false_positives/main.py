import sys
sys.path.append('../../model')

import os

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

    Classifier.classify_files(h=h, j=j, v=v, z=z, out_dir=out_dir)

if __name__=='__main__':
    main()


