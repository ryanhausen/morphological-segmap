import sys
sys.path.append('../model')

import itertools
import os
from multiprocessing import Pool
from shutil import rmtree

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from astropy.io import fits
from scipy.stats import entropy
from sklearn.metrics import confusion_matrix

from inference import Classifier
from class_extractor import get_classification

def safe_fits(fits_file):
    f = fits.getdata(fits_file)
    img = f.copy()
    del f
    return img

def agreement(arr):
    return 1 - entropy(arr, base=2) / np.log2(len(arr))


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    vmin, vmax= cm.min(), cm.max()
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        vmin, vmax = 0, 1



    plt.imshow(cm, interpolation='nearest', cmap=cmap, vmin=vmin, vmax=vmax)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


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
    srcs, labels, predictions, agreements = [], [], [], []

    output_dir_mask = src_dir + '/{}/output'

    for src in tqdm(sorted(os.listdir(src_dir)), desc='Aggregating'):
        srcs.append(src)
        output_dir = output_dir_mask.format(src)

        with open(os.path.join(src_dir, src, 'label'), 'r') as f:
            theirs = np.array(f.readline().split(',')[2:]).astype(np.float32)
        labels.append(theirs)
        agreements.append(agreement(theirs))

        morphs = ['spheroid', 'disk', 'irregular', 'point_source', 'background']
        data = {}

        xs = ys = slice(5,-5)
        for m in morphs:
            m_path = os.path.join(output_dir, 'top_{}.fits'.format(m))
            data[m] = safe_fits(m_path)[ys, xs]

        # H Band Flux
        flux = safe_fits(os.path.join(src_dir, src, 'data.fits'))[0, ys, xs]

        ours = get_classification(flux,
                                  data['spheroid'],
                                  data['disk'],
                                  data['irregular'],
                                  data['point_source'],
                                  data['background'])
        np.savetxt(os.path.join(src_dir, src, 'ours.npy'), ours)

        predictions.append(ours)

    with open('outputs.csv', 'w') as f:
        f.write('ID,l_sph,l_dk,l_irr,l_ps,l_unk,sph,dk,irr,ps,bkg,agg\n')
        for s, l, p, a in tqdm(zip(srcs, labels, predictions, agreements),
                               desc='writing'):

            line = s + ','
            line += ','.join([str(v) for v in l]) + ','
            line += ','.join([str(v) for v in p]) + ','
            line += str(a) + '\n'

            f.write(line)

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

def make_figs():
    lbls, preds = [], []
    with open('outputs.csv', 'r') as f:
        for line in f.readlines()[1:]:
            line = line.strip().split(',')
            if float(line[-1]) > 0.98:
                theirs = np.array([float(v) for v in line[1:6]])
                ours = np.array([float(v) for v in line[6:-1]])
                if theirs.argmax() < 4:
                    lbls.append(theirs.argmax())
                    preds.append(ours.argmax())

    labels = ['Spheroid', 'Disk', 'Irregular', 'Point Source']
    plt.figure(figsize=(10,10))
    plot_confusion_matrix(confusion_matrix(lbls, preds), labels, normalize=True)
    plt.savefig('normed_confusion.pdf', dpi=600)




def main(remove_prev_classifications=True,
         classify_pixel=True,
         aggregate_classification=True,
         output_figs=True):
    img_dir = '../data/imgs'
    if remove_prev_classifications:
        delete_classications(img_dir)

    if classify_pixel:
        classify_pixels(img_dir)

    if aggregate_classification:
        aggregate_classifications(img_dir)

    if output_figs:
        make_figs()

if __name__=='__main__':
    main(remove_prev_classifications=False,
         classify_pixel=False,
         aggregate_classification=True)
