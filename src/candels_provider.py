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
                 batch_size=25,
                 overlap_ratio=0.5,
                 limiting_range=None,
                 labels_file='labels.csv',
                 use_weighted_labels=False,
                 data_format='channels_first'):
        self.data_dir = data_dir
        self.split = split
        self.batch_size = batch_size
        self.input_size = input_size
        self.area_size = area_size
        self.use_weighted_labels = use_weighted_labels
        self.data_format = data_format
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
            header = fits.getheader(os.path.join(data_dir, img_fits_files[0]))
            start_y, end_y = 0, header['NAXIS2']
            start_x, end_x = 0, header['NAXIS1']
            del header

        train_data = []
        test_data = []

        step_y = int(area_size[0] * (1 - overlap_ratio))
        step_x = int(area_size[1] * (1 - overlap_ratio))
        split_factor = int((end_y - start_y) * split)
        for i in range(start_y, start_y+split_factor, step_y):
            for j in range(start_x, end_x, step_x):
                train_data.append((i, j))

        for i in range(start_y+split_factor, end_y, input_size[0]):
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
            self._train = self._train.map(self.preprocess_train)

            if self.data_format == 'channels_first':
                self._train = self._train.map(self.transpose_dataset)

            training_data = self._train.shuffle(self.batch_size*10)
            training_data = self._train.repeat(self.NUM_REPEAT)
            training_data = self._train.batch(self.batch_size)
            self._train_iter = training_data.make_one_shot_iterator()

        x, y, w = self._train_iter.get_next()

        return x, y, w

    @property
    def test(self):
        if self._test is None:
            test_data = self.test_data.map(self.tf_test_img_from_idx)
            test_data = test_data.map(self.preprocess_test)
            if self.data_format == 'channels_first':
                test_data = test_data.map(self.transpose_dataset)

            test_data = test_data.batch(self.batch_size)
            self._test = test_data
            self._test_iter = self._test.make_one_shot_iterator()

        x, y, w = self._test_iter.get_next()

        return x, y, w


    def tf_train_img_from_idx(self, idx):
        i, j = idx[0], idx[1]
        return self._tf_img_from_idx(i, j, True)

    def tf_test_img_from_idx(self, idx):
        i, j = idx[0], idx[1]
        return self._tf_img_from_idx(i, j, False)

    def preprocess_train(self, x, y, w):
        return self._preprocess_data(x, y, w, True)

    def preprocess_test(self, x, y, w):
        return self._preprocess_data(x, y, w, False)

    def transpose_dataset(self, x, y, w):
        x = tf.transpose(x, perm=[2, 0, 1])
        y = tf.transpose(y, perm=[2, 0, 1])
        w = tf.transpose(w, perm=[2, 0, 1])

        return x, y, w

    def _preprocess_data(self, x, y, w, is_training):
        with tf.name_scope('preprocessing'):
            if is_training:
                input_channels = len(self.data)-1
                total_channels = input_channels + self.NUM_CLASSES + 1

                t = tf.concat([x, y, w], axis=2)
                t = tf.random_crop(t, [self.input_size[0],
                                       self.input_size[1],
                                       total_channels])

                t = self._augment(t)

                x = t[:, :, :input_channels]
                y = t[:, :, input_channels:input_channels+self.NUM_CLASSES]
                w = t[:, :, -1:]

            x = self._standardize(x)

            return x, y, w


    def _tf_img_from_idx(self, i, j, is_training):
        func_inputs = [i, j, is_training]
        func_output_types = (tf.float32, tf.float32, tf.float32)

        x, y, w = tf.py_func(self._imgs_from_idx,
                             func_inputs,
                             func_output_types)

        if is_training:
            shape = [self.area_size[0], self.area_size[1]]
        else:
            shape = [self.input_size[0], self.input_size[1]]

        x.set_shape(shape + [len(self.data)-1])
        y.set_shape(shape + [DataProvider.NUM_CLASSES])
        w.set_shape(shape + [1])

        return x, y, w


    def _imgs_from_idx(self, i, j, is_training):
        i_pad, j_pad = self.area_size if is_training else self.input_size

        x =[d[i:i+i_pad, j:j+j_pad] for d in self.data[:-1]]
        segmap = self.data[-1][i:i+i_pad, j:j+j_pad]
        weight_map = np.zeros_like(segmap, dtype=np.float32)

        label_map = self._get_label_map(segmap)

        x = np.dstack(x).astype(np.float32)

        y = np.zeros([i_pad, j_pad, DataProvider.NUM_CLASSES],
                        dtype=np.float32)

        for lbl in label_map:
            mask = segmap==lbl
            lbl_probs = label_map[lbl]
            y[mask, :] = lbl_probs
            weight_map[mask] = lbl_probs.sum() > 0

        return (x, y, weight_map[:, :, np.newaxis].astype(np.float32))

    def _get_label_map(self, segmap):
        ids = np.unique(segmap)

        if self.use_weighted_labels:
            lbl_cols = DataProvider.WEIGHTED_LABELS_COLS
        else:
            lbl_cols = DataProvider.LABELS_COLS

        lbl_map = {0:DataProvider.BACKGROUND}
        for i in ids:
            if i > 0:
                rows = self.raw_labels['ID']==i
                vals = self.raw_labels.loc[rows, lbl_cols].values.flatten()
                if len(vals)==0:
                    vals = np.zeros(DataProvider.NUM_CLASSES - 1)
                # insert 0 for background
                lbl_map[i] = np.insert(vals, 4, 0).astype(np.float32)

        return lbl_map

    def _augment(self, t):
        with tf.name_scope('augmentation'):
            angle = tf.random_uniform([1], maxval=360)
            t = tf.contrib.image.rotate(t, angle, interpolation='NEAREST')
            t = tf.image.random_flip_left_right(t)
            t = tf.image.random_flip_up_down(t)

        return t

    def _standardize(self, x):
        with tf.name_scope('image_standardization'):
            x = tf.image.per_image_standardization(x)
            x = tf.reduce_mean(x, axis=-1, keepdims=True)

        return x

def main():
    data_dir = '../data'

    input_files = [
                    'hlsp_candels_hst_wfc3_gs-tot_f160w_v1.0_drz.fits'
                  ]
    segmap_file = 'segmap.fits'

    dataset = DataProvider(input_files,
                           segmap_file,
                           batch_size=25,
                           limiting_range=[(16000, 25000), (10500, 20000)])
    expected_x = (dataset.batch_size,
                  1,
                  dataset.input_size[0],
                  dataset.input_size[1])
    expected_y = (dataset.batch_size,
                  DataProvider.NUM_CLASSES,
                  dataset.input_size[0],
                  dataset.input_size[1])
    expected_w = (dataset.batch_size,
                  1,
                  dataset.input_size[0],
                  dataset.input_size[1])

    with tf.Session() as sess:

        print('Asserting train shape')
        x, y, w = sess.run(dataset.train)
        assert x.shape==expected_x
        assert y.shape==expected_y
        assert w.shape==expected_w

        print('Asserting test shape')
        x, y, w = sess.run(dataset.test)
        assert x.shape==expected_x
        assert y.shape==expected_y
        assert w.shape==expected_w


if __name__=='__main__':
    main()

