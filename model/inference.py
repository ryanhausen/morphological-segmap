import time
from collections import namedtuple

from astropy.io import fits
import tensorflow as tf

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
        print('Making {}...'.format(f_name))
        stub = np.zeros([100,100], dtype=dtype)

        hdu = fits.PrimaryHDU(data=stub)
        header = hdu.header
        while len(header) < (36 * 4 - 1):
            header.append()
        header['NAXIS1'] = shape[1]
        header['NAXIS2'] = shape[0]
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
            raise Exception('Didn\'t assign bytes_per_value')

        with open(f_name, 'rb+') as f:
            f.seek(len(header.tostring()) + (naxis1 * naxis2 * bytes_per_value) - 1)
            f.write(b'\0')

    @staticmethod
    def _prepare_out_files(bands, out_dir, out_type):
        mean_var = []
        ranking = []
        for m in Classifier.MORPHOLOGIES:
            mean_var.extend(['{}_mean.fits'.format(m), '{}_var.fits'.format(m)])
            ranking.append('{}.fits'.format(m))








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
        h, j, v, z = Classifier._validate_arrays(h, j, v, z)








