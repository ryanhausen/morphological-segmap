import os

import numpy as np
import pandas as pd
import tensorflow as tf
from astropy.io import fits

class DataProvider:
    """

    Labels:
    [Smooth, Features, Disk, Star/Artifact, Background,
     Round, Inbetween, Cigar, Clumpy]
    """
    DATA_FORMAT = 'channels_first' # or channels_last
    DATA_FORMAT_ERR = ValueError('Invalid setting for DataProvider.DATA_FORMAT')
    NUM_REPEAT = 1

    NUM_CLASSES = 9
    BACKGROUND = np.array([0,0,0,0,1,0,0,0,0], dtype=np.float32)

    LABELS_COLS = ['SMOOTH', 'FEATURES_DISK', 'STAR_ARTIFACT',
                   'ROUND', 'IN_BETWEEN', 'CLUMPY']

    WEIGHTED_LABELS_COLS = ['W_SMOOTH', 'W_FEATURES_DISK', 'W_STAR_ARTIFACT',
                            'W_ROUND', 'W_IN_BETWEEN', 'W_CLUMPY']

    def __init__(self,
                 img_fits_files,
                 segnap_fits_file,
                 data_dir='../data',
                 split=0.8,
                 input_size=(100, 100),
                 area_size=(250, 250),
                 samples_per_area=25,
                 batch_size=1,
                 overlap_ratio=0.5,
                 limiting_range=None,
                 labels_file='labels.csv',
                 use_weighted_labels=False):
        self.data_dir = data_dir
        self.split = split
        self.batch_size = batch_size
        self.samples_per_area = samples_per_area
        self.input_size = input_size
        self.area_size = area_size
        self.use_weighted_labels = use_weighted_labels
        self.raw_labels = pd.read_csv(os.path.join(data_dir, labels_file))
        self.hduls = []
        self.data = []

        for i in img_fits_files + [segnap_fits_file]:
            hdul = fits.open(os.path.join(data_dir, i),
                             memmap=True,
                             mode='readonly')
            self.hduls.append(hdul)
            self.data.append(hdul[0].data)

        if limiting_range:
            start_y, end_y = limiting_range[0]
            start_x, end_x = limiting_range[1]
        else:
            header = fits.getheader(imgs[0])
            start_y, end_y = 0, header['NAXIS2']
            start_x, end_x = 0, header['NAXIS1']
            del header

        train_data = []
        test_data = []

        step_y = area_size[0] * (1 - overlap_ratio)
        step_x = area_size[1] * (1 - overlap_ratio)
        for i in range(start_y, int(end_y * split - area_size[0]), step_y):
            for j in range(start_x, end_x, step_x):
                train_data.append((i, j))

        for i in range(int(end_y * split), end_y, input_size[0]):
            for j in range(start_x, end_x, input_size[1]):
                test_data.append((i, j))

        self.train_data = tf.data.Dataset.from_tensor_slices(train_data)
        self.test_data = tf.data.Dataset.from_tensor_slices(test_data)

        self._train = None
        self._test = None

    @property
    def train(self):
        if self._train is None:
            self._train = self.train_data.map(self.tf_train_img_from_idx)

    def tf_train_img_from_idx(self, i, j):
        func_inputs = [i, j, True]
        func_output_types = (tf.float32, tf.float32, tf.float32)

        x, y = tf.py_func(self._imgs_from_idx,
                          func_inputs,
                          func_output_types)

        shape = [self.samples_per_area, self.input_size[0], self.input_size[1]]

        if DataProvider.DATA_FORMAT=='channels_first':
            x.set_shape(np.insert(shape, 1, len(self.data)))
            y.set_shape(np.insert(shape, 1, DataProvider.NUM_CLASSES))
        elif DataProvider.DATA_FORMAT=='channels_last':
            x.set_shape(shape + [len(self.data)])
            y.set_shape(shape + [DataProvider.NUM_CLASSES])
        else:
            raise DataProvider.DATA_FORMAT_ERR

        return x, y

    def tf_test_img_from_idx(self, i, j):
        func_inputs = [i, j, False]
        func_output_types = (tf.float32, tf.float32, tf.float32)

        x, y = tf.py_func(self._imgs_from_idx,
                          func_inputs,
                          func_output_types)

        shape = [self.samples_per_area, self.input_size[0], self.input_size[1]]

        if DataProvider.DATA_FORMAT=='channels_first':
            x.set_shape(np.insert(shape, 1, len(self.data)))
            y.set_shape(np.insert(shape, 1, DataProvider.NUM_CLASSES))
        elif DataProvider.DATA_FORMAT=='channels_last':
            x.set_shape(shape + [len(self.data)])
            y.set_shape(shape + [DataProvider.NUM_CLASSES])
        else:
            raise DataProvider.DATA_FORMAT_ERR

        return x, y


    def _imgs_from_idx(self, i, j, is_training):
        i_pad, j_pad = self.area_size if is_training else self.input_size

        x =[d[i:i+i_pad, j:j+j_pad] for d in self.data[:-1]]
        segmap = self.data[-1][i:i+i_pad, j:j+j_pad]

        label_map = self._get_label_map(segmap)

        if DataProvider.DATA_FORMAT=='channels_first':
            x = np.array(x)
            y = np.zeros([DataProvider.NUM_CLASSES, i_pad, j_pad],
                         dtype=np.float32)

            for lbl in label_map:
                mask = segmap==lbl
                y[:,mask] = label_map[lbl]

        elif DataProvider.DATA_FORMAT=='channels_last':
            x = np.dstack(x)
            y = np.zeros([i_pad, j_pad, DataProvider.NUM_CLASSES],
                         dtype=np.float32)

            for lbl in label_map:
                mask = segmap==lbl
                y[mask, :] = label_map[lbl]
        else:
            raise DataProvider.DATA_FORMAT_ERR

        return x, y


    def _get_label_map(self, segmap):
        ids = np.unique(segmap)

        if self.use_weighted_labels:
            lbl_cols = DataProvider.WEIGHTED_LABELS_COLS
        else:
            lbl_cols = DataProvider.LABELS_COLS

        lbl_map = {0:DataProvider.BACKGROUND}
        for i in ids:
            if i > 0:
                row_filter = self.raw_labels['ID']==i
                vals = self.raw_labels[row_filter, lbl_cols].values.flatten()
                # insert 0 for background
                lbl_map[i] = np.insert(vals, 4, 0).astype(np.float32)

        return lbl_map


    def _random_crop(self, img, num):
        None