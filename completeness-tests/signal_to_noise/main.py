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

    if 'output' not in os.listdir():
        os.mkdir('./output')

    # create or clear directories and run tests
    for test_file in os.listdir('./tests'):
        out_dir = './output/{}'.format(test_file.replace('.fits', ''))
        if out_dir not in os.listdir('./tests'):
            os.mkdir(out_dir)
        else:
            for f in os.listdir(out_dir):
                os.remove(os.path.join(out_dir, f))

        test = fits.getdata(os.path.join('./tests', test_file))

        h = test[0, :, :]
        j = test[1, :, :]
        v = test[2, :, :]
        z = test[3, :, :]

        Classifier.classify_arrays(h=h, j=j, v=v, z=z, out_dir=out_dir)

if __name__=='__main__':
    main()


