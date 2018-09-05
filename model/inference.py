import json
import os
import time
from collections import namedtuple

import numpy as np
from astropy.io import fits
import tensorflow as tf
from tqdm import tqdm

from model import Model

class Classifier:
    __graph = None
    __session = None
    X = tf.placeholder(tf.float32, shape=[None, 40, 40, 4])
    DATASET = namedtuple('Dataset', ['num_labels'])
    MORPHOLOGIES = [
        'spheroid',
        'disk',
        'irregular',
        'point_source',
        'background'
    ]
    UPDATE_MASK = np.pad(np.ones([30,30]), 5, mode='constant').astype(np.int16)
    N_UPDATE = np.ones([40, 40], dtype=np.int16)

    @staticmethod
    def classify_files(h=None,
                       j=None,
                       v=None,
                       z=None,
                       out_dir='.',
                       batch_size=1000,
                       out_type='both'):
        h, j, v, z = Classifier._validate_files(h, j, v, z)

        Classifier.classify_arrays(h=h,
                                   j=j,
                                   v=v,
                                   z=z,
                                   out_dir=out_dir,
                                   batch_size=batch_size)

    @staticmethod
    def classify_arrays(h=None,
                        j=None,
                        v=None,
                        z=None,
                        out_dir='.',
                        batch_size=1000,
                        out_type='both'):
        bands = Classifier._validate_arrays(h, j, v, z)

        shape = bands[0].shape
        hduls, data = Classifier._prepare_out_files(shape, out_dir, out_type)

        window_y, window_x = Classifier.N_UPDATE.shape
        final_y = shape[0] - window_y
        final_x = shape[1] - window_x

        index_gen = Classifier._index_generator(final_y+1, final_x+1)

        num_batches = final_y * final_x // batch_size

        for _ in tqdm(range(num_batches+1), desc='Classifying', unit='batch'):
            batch = []
            batch_idx = []

            for _ in range(batch_size):
                try:
                    y, x = next(index_gen)
                except StopIteration:
                    break
                combined = [b[y:y+window_y, x:x+window_x] for b in bands]
                batch.append(Classifier._preprocess(np.array(combined)))
                batch_idx.append((y, x))

            batch = np.array(batch)

            labels = Classifier._run_model(batch)

            Classifier._update_outputs(data, labels, batch_idx, out_type)

        for hdul in hduls:
            hdul.close()

    @staticmethod
    def _variables_not_none(names, values):
        nones = []
        for name, value in zip(names, values):
            if value is None:
                nones.append(name)

        if len(nones) > 0:
            raise ValueError('{} should not be None'.format(nones))

    @staticmethod
    def _validate_files(h, j, v, z):
        Classifier._variables_not_none('hjvz', [h, j, v, z])

        arrays = []
        for f in [h, j, v, z]:
            hdul = fits.open(f, mode='readonly', memmap=True)
            arrays.append(hdul[0].data)
            hdul.close()

        h, j, v, z = arrays

        return Classifier._validate_arrays(h, j, v, z)

    @staticmethod
    def _validate_arrays(h, j, v, z):
        Classifier._variables_not_none('hjvz', [h, j, v, z])

        ax0, ax1 = h.shape
        for arr in [j, v, z]:
            a0, a1 = arr.shape
            if a0 != ax0:
                raise ValueError('Dimension mismatch {} != {}'.format(ax0, a0))

            if a1 != ax1:
                raise ValueError('Dimension mismatch {} != {}'.format(ax1, a1))

        return [h, j, v, z]

    # http://docs.astropy.org/en/stable/generated/examples/io/skip_create-large-fits.html
    @staticmethod
    def _create_file(f_name, shape, dtype):
        stub_size = [100, 100]
        if len(shape)==3:
            stub_size.append(5)
        stub = np.zeros(stub_size, dtype=dtype)

        hdu = fits.PrimaryHDU(data=stub)
        header = hdu.header
        while len(header) < (36 * 4 - 1):
            header.append()

        header['NAXIS1'] = shape[1]
        header['NAXIS2'] = shape[0]
        if len(shape)==3:
            header['NAXIS3'] = shape[2]

        header.tofile(f_name)

        bytes_per_value = 0

        if dtype==np.uint8:
            bytes_per_value = 1
        elif dtype==np.int16:
            bytes_per_value = 2
        elif dtype==np.float32:
            bytes_per_value = 4
        elif dtype==np.float64:
            bytes_per_value = 8

        if bytes_per_value==0:
            raise ValueError('Invalid dtype')

        with open(f_name, 'rb+') as f:
            header_size = len(header.tostring())
            data_size = (np.prod(shape) * bytes_per_value) - 1

            f.seek(header_size + data_size)
            f.write(b'\0')

    @staticmethod
    def _prepare_out_files(shape, out_dir, out_type):
        with_out_dir = lambda f: os.path.join(out_dir, '{}.fits'.format(f))

        mean_var = []
        ranking = []
        for m in Classifier.MORPHOLOGIES:
            mean_var.append('{}_mean'.format(m))
            mean_var.append('{}_var'.format(m))
            ranking.append('{}'.format(m))
        n = 'n'

        hduls = []
        data = {}

        if out_type in ['mean_var', 'both']:
            for f in mean_var:
                _f = with_out_dir(f)
                Classifier._create_file(_f, shape, np.float32)
                hdul = fits.open(_f, mode='update', memmap=True)
                hduls.append(hdul)
                data[f] = hdul[0].data


        if out_type in ['rank_vote', 'both']:
            for f in ranking:
                _f = with_out_dir(f)
                Classifier._create_file(_f, list(shape) + [5], np.float32)
                hdul = fits.open(_f, mode='update', memmap=True)
                hduls.append(hdul)
                data[f] = hdul[0].data



        _n = with_out_dir(n)
        Classifier._create_file(_n, shape, np.int16)
        hdul = fits.open(_n, mode='update', memmap=True)
        data[n] = hdul[0].data

        return hduls, data

    @staticmethod
    def _index_generator(upto_y, upto_x):
        for y in range(upto_y):
            for x in range(upto_x):
                yield (y, x)

    # https://www.tensorflow.org/api_docs/python/tf/image/per_image_standardization
    @staticmethod
    def _preprocess(img):
        num = img - img.mean()
        denom = max(img.std(), 1/np.sqrt(np.prod(img.shape)))
        return  num / denom

    @staticmethod
    def _run_model(batch):
        batch = np.transpose(batch, axes=[0, 2, 3, 1])

        if Classifier.__graph is None:
            get_local = lambda f: os.path.join(os.path.dirname(__file__), f)

            model_file = get_local('model_config.json')
            with open(model_file, 'r') as f:
                model_params = tf.contrib.training.HParams(**json.load(f))

            dataset = Classifier.DATASET(5)
            model = Model(model_params, dataset, 'channels_last')
            Classifier.__graph = model.inference(Classifier.X)
            saver = tf.train.Saver()
            Classifier.__session = tf.Session()

            weights_dir = get_local('model_weights')
            saver.restore(Classifier.__session,
                          tf.train.latest_checkpoint(weights_dir))

        return Classifier.__session.run(Classifier.__graph,
                                        feed_dict={Classifier.X:batch})

    @staticmethod
    def _get_final_map(shape, y, x):
        final_map = []

        end_y = y==(shape[0] - Classifier.N_UPDATE.shape[0])
        end_x = x==(shape[1] - Classifier.N_UPDATE.shape[1])

        if end_y and end_x:
            for _y in range(0, 35):
                for _x in range(5, 35):
                    final_map.append((_y, _x))
        else:
            if end_x:
                final_map.extend([(5,_x) for _x in range(5, 35)])
            if end_y:
                final_map.extend([(_y,5) for _y in range(5, 35)])

        if len(final_map)==0:
            final_map.append((5, 5))

        return final_map

    # http://people.ds.cam.ac.uk/fanf2/hermes/doc/antiforgery/stats.pdf, eq. 4
    @staticmethod
    def _iterative_mean(n, prev_mean, x_n, update_mask):
        n[n==0] = 1
        return prev_mean + ((x_n - prev_mean)/n * update_mask)

    # http://people.ds.cam.ac.uk/fanf2/hermes/doc/antiforgery/stats.pdf, eq. 24
    @staticmethod
    def _iterative_variance(n, prev_var, x_n, prev_mean, curr_mean, update_mask):
        return prev_var + ((x_n - prev_mean) * (x_n - curr_mean) * update_mask)

    @staticmethod
    def _finalize_variance(n, curr_var, final_map):
        final_n = np.ones_like(n)
        for y,x in final_map:
            final_n[y,x] = n[y,x]

        return curr_var / final_n

    @staticmethod
    def _updated_mv(n, x_n, prev_mean, prev_var, update_mask, final_map):
        curr_mean = Classifier._iterative_mean(n, prev_mean, x_n, update_mask)

        curr_var = Classifier._iterative_variance(n,
                                                  prev_var,
                                                  x_n,
                                                  prev_mean,
                                                  curr_mean,
                                                  update_mask)

        curr_var = Classifier._finalize_variance(n, curr_var, final_map)

        return curr_mean, curr_var

    @staticmethod
    def _updated_count(n, x_n, prev_count, update_mask, final_map):
        update = np.zeros_like(prev_count)

        for i in range(update.shape[1]):
            for j in range(update.shape[2]):
                if update_mask[i,j]:
                    update[x_n[i,j],i,j] = 1

        count = prev_count + update

        return count
        # final_n = np.ones_like(n)
        # for y,x in final_map:
        #     final_n[y, x] = n[y, x]

        # return count / final_n

    @staticmethod
    def _update_outputs(data, labels, batch_idx, out_type):

        window_y, window_x = Classifier.N_UPDATE.shape
        for i, l in enumerate(labels):
            y, x = batch_idx[i]
            ys = slice(y, y+window_y)
            xs = slice(x, x+window_x)

            ns = data['n'][ys, xs]
            ns = ns + (Classifier.N_UPDATE * Classifier.UPDATE_MASK)
            data['n'][ys, xs] = ns

            final_map = Classifier._get_final_map(data['n'].shape, y, x)
            if out_type in ['mean_var', 'both']:
                for j, morph in enumerate(Classifier.MORPHOLOGIES):
                    k_mean = '{}_mean'.format(morph)
                    k_var = '{}_var'.format(morph)

                    x_n = l[:, :, j]
                    prev_mean = data[k_mean][ys, xs]
                    prev_var = data[k_var][ys, xs]

                    mean, var = Classifier._updated_mv(ns,
                                                       x_n,
                                                       prev_mean,
                                                       prev_var,
                                                       Classifier.UPDATE_MASK,
                                                       final_map)

                    data[k_mean][ys, xs] = mean
                    data[k_var][ys, xs] = var

            if  out_type in ['rank_vote', 'both']:
                ranked = l.argsort().argsort()

                for j, morph in enumerate(Classifier.MORPHOLOGIES):
                    # TODO: figure out why the dims are transposed here
                    prev_count = data[morph][:, ys, xs]

                    count = Classifier._updated_count(ns,
                                                      ranked[:, :, j],
                                                      prev_count,
                                                      Classifier.UPDATE_MASK,
                                                      final_map)

                    data[morph][:, ys, xs] = count

