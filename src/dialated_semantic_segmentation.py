"""
https://arxiv.org/pdf/1804.04020.pdf
"""
import types

import tensorflow as tf
layers = tf.layers
var_init = tf.variance_scaling_initializer

import model
from tf_src import tf_logger as log

class Model(model.Model):
    KERNEL_SETTINGS = [
        (5, 32, 1),
        (5, 32, 2),
        (4, 64, 3),
        (4, 64, 4),
        (3, 128, 5),
        (3, 128, 6)
    ]
    DATA_FORMAT = 'channels_first'
    NUM_CLASSES = 5

    def __init__(self, dataset, is_training):
        model.Model.DATA_FORMAT = Model.DATA_FORMAT
        super().__init__(dataset, is_training)

    def model_fn(self, x):
        weight_decay = 1e-2
        regularizer = tf.contrib.layers.l2_regularizer(scale=weight_decay)

        concat_axis = 1 if Model.DATA_FORMAT=='channels_first' else 3
        outputs = []

        for i, (k, filters, d) in enumerate(Model.KERNEL_SETTINGS):
            with tf.variable_scope('layer_{}'.format(i)):
                if len(outputs) > 1:
                    x = tf.concat(outputs,
                                concat_axis,
                                name='concat_{}'.format(i))

                log.tensor_shape(x)

                x = layers.conv2d(x,
                                filters,
                                k,
                                padding='SAME',
                                data_format=Model.DATA_FORMAT,
                                dilation_rate=d,
                                kernel_initializer=var_init,
                                kernel_regularizer=regularizer,
                                activation=tf.nn.relu,
                                name='cov2d_{}'.format(i))
                log.tensor_shape(x)

                outputs.append(x)

        with tf.name_scope('classification_layer'):
            x = tf.concat(outputs, concat_axis, name='output_concat')
            log.tensor_shape(x)

            logits = layers.conv2d(x,
                                Model.NUM_CLASSES,
                                1,
                                padding='SAME',
                                data_format=Model.DATA_FORMAT,
                                kernel_initializer=var_init,
                                kernel_regularizer=regularizer,
                                activation=None,
                                name='output_conv2d')
            log.tensor_shape(logits)

        return logits


def main():
    info = """
    CANDELS Morphological Classification -- Semantic Segmentation
    Dialated ConvNet
    DATA_FORMAT:    {}
    """.format(Model.DATA_FORMAT)

    tf.logging.set_verbosity(tf.logging.DEBUG)
    print(info)

    in_shape = [5,1,40,40]
    expect_out_shape = [5,5,40,40]

    x = tf.placeholder(tf.float32, shape=in_shape, name='input')

    mock_dataset = types.SimpleNamespace(NUM_LABELS=5)
    Model.DATA_FORMAT = 'channels_first'
    m = Model(mock_dataset, True)

    out = m.build_graph(x)

    assert out.shape.as_list()==expect_out_shape, "Incorrect Shape"

if __name__=='__main__':
    main()