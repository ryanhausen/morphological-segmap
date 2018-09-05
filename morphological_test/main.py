import sys
sys.path.append('../model')

import os
from multiprocessing import Pool
from shutil import rmtree

import numpy as np
from tqdm import tqdm
from astropy.io import fits

from inference import Classifier
from class_extractor import get_classification

def safe_fits(fits_file):
    f = fits.getdata(fits_file)
    img = f.copy()
    del f
    return img


def delete_classications(classification_dir):
    fits_to_delete = [
        'background',
        'disk',
        'irregular',
        'n',
        'point_source',
        'spheroid',
    ]
    other_to_delete  = ['ours']
    dir_to_delete = ['post-processing', 'output']

    for src in tqdm(sorted(os.listdir(classification_dir)),
                    desc='Deleting',
                    unit='src'):
        src_dir = os.path.join(classification_dir, src)

        for f in fits_to_delete:
            try:
                if os.path.exists(os.path.join(src_dir, '{}.fits'.format(f))):
                    os.remove(os.path.join(src_dir, '{}.fits'.format(f)))
            except Exception:
                pass

        for f in other_to_delete:
            try:
                if os.path.exists(os.path.join(src_dir, f)):
                    os.remove(os.path.join(src_dir, f))
            except Exception:
                pass

        for d in dir_to_delete:
            try:
                if os.path.exists(os.path.join(src_dir, d)):
                    rmtree(os.path.join(src_dir, d))
            except Exception:
                pass

def aggregate_classifications(src_dir):
    correct, num, agg = [], [], []

    output_dir_mask = src_dir + '/{}/output'

    for src in os.listdir(src_dir):
        output_dir = output_dir_mask.format(src)

        with open(os.path.join(src_dir, src, 'label'), 'r') as f:
            theirs = np.array(f.readline().split(',')[2:]).astype(np.float32)

        morphs = ['spheroid', 'disk', 'irregular', 'point_source', 'background']

        data = {}

        for m in morphs:
            data[m] = safe_fits(os.path.join(output_dir, 'top_{}.fits'.format(m)))













def classify_img_dir(img_dir):
    data = fits.getdata(os.path.join(img_dir, 'data.fits'))

    h, j, v, z = [data[i,:,:] for i in range(4)]

    out_dir = os.path.join(img_dir, 'output')
    if 'output' not in os.listdir(img_dir):
        os.mkdir(out_dir)

    Classifier.classify_arrays(h=h,
                               j=j,
                               v=v,
                               z=z,
                               out_dir=out_dir,
                               batch_size=1000)

def classify_pixels(classification_dir, num_processes=2):

    img_dirs = [os.path.join(classification_dir, d) for d in os.listdir(classification_dir)]

    with Pool(num_processes) as p:
        list(tqdm(p.imap(classify_img_dir, img_dirs), total=len(img_dirs)))

def main(remove_prev_classifications=True,
         classify_pixel=True,
         aggregate_classification=True):
    img_dir = '../data/imgs'
    if remove_prev_classifications:
        delete_classications(img_dir)

    if classify_pixel:
        classify_pixels(img_dir)

    if aggregate_classification:
        aggregate_classifications(img_dir)

if __name__=='__main__':
    main(remove_prev_classifications=False,
         classify_pixel=False,
         aggregate_classification=True)