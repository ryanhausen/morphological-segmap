import sys
sys.path.append('../model')

import os

import numpy as np
from tqdm import tqdm

from inference import Classifier

def main():
    data_dir = '../data/GOODS-Garth'
    output_dir = './output'

    if 'output' not in os.listdir():
        os.mkdir('./output')

    files = {
        'h':os.path.join(data_dir, 'h_f160w.fits'),
        'j':os.path.join(data_dir, 'j_f125w.fits'),
        'v':os.path.join(data_dir, 'v_f606w.fits'),
        'z':os.path.join(data_dir, 'z_f850lp.fits')
    }

    Classifier.classify_files(h=files['h'],
                              j=files['j'],
                              v=files['v'],
                              z=files['z'],
                              batch_size=2000,
                              out_dir=output_dir,
                              paralell_gpus=True)


if __name__=='__main__':
    main()


