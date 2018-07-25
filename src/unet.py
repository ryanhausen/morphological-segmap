import types
import tensorflow as tf
layers =  tf.layers


var_init = tf.variance_scaling_initializer


class Model:
    NAME = 'convnet_semantic_segmentation'
    DATA_FORMAT = 'channels_first'
    FILTER_COUNT = [64, 32, 16]

    def __init__(self, dataset, is_training):
        self.dataset = dataset
        self.is_training = is_training

        self._graph = None
        self._train = None
        self._optimizer = None
        self._test = None
        self._inference = None
        self._loss_func = None
        self._train_metrics = None
        self._test_metrics = None

    def build_graph(self, x, output_pyramid=False):
        """
        Builds the graph.
        INPUTS:
        - x: input image tensor, given by dataset, in train() or test()
        - output_pyramid: boolean, if true produces output at each upconv scaled
                          resolution if false, only returns the output of at the
                          final upscaled resolution.
        RETURNS:
        - a single tensor of [n, w, h, NUM_LABELS] if output_pyramid=True, else
          returns a list of tensors of m [n, w/r, h/r, NUM_LABELS] m tensors
          where r is the reduction factor associated with the downsample size at
          that depth in the network.
        """


        if self._graph:
            return self._graph(x)

        label_pyramid = []
        concat_axis = 1 if Model.DATA_FORMAT=='channels_first' else 3

        def model_fn(inputs):
            outputs = []

            for i, f in enumerate(Model.FILTER_COUNT):
                with tf.variable_scope('downconv-{}'.format(i)):
                    inputs = Model.conv(inputs, f)
                    outputs.append(inputs)
                    inputs = Model.down_sample(inputs)

            with tf.variable_scope('intermediary_conv'):
                inputs = Model.conv(inputs, Model.FILTER_COUNT[-1]//2)

            for i, f in enumerate(reversed(Model.FILTER_COUNT)):
                with tf.variable_scope('upconv-{}'.format(i)):
                    inputs = Model.up_sample(inputs)
                    inputs = tf.concat([inputs, outputs[-(i+1)]],
                                       concat_axis,
                                       name='concatenated')
                    inputs = Model.conv(inputs, f)
                    if output_pyramid:
                        label_pyramid.append(Model.conv(inputs,
                                                        self.dataset.NUM_CLASSES,
                                                        activation=None,
                                                        name='pyramid_conv'))


            with tf.variable_scope('out_conv'):
                inputs = Model.conv(inputs,
                                    self.dataset.NUM_CLASSES,
                                    activation=None,
                                    name='conv')

            if output_pyramid:
                return label_pyramid + [inputs]
            else:
                return inputs

        self._graph = model_fn
        return self._graph(x)

    @staticmethod
    def conv(x,
             num_filters,
             padding='same',
             strides=1,
             activation=tf.nn.elu,
             name='downconv'):
        kernel_size = 3
        x = layers.conv2d(x,
                          num_filters,
                          kernel_size,
                          padding=padding,
                          strides=strides,
                          kernel_initializer=var_init,
                          use_bias=True,
                          data_format=Model.DATA_FORMAT,
                          name=name)

        if activation:
            return tf.check_numerics(activation(x), 'numeric check failed')
        else:
            return tf.check_numerics(x, 'numeric check failed')

    @staticmethod
    def down_sample(x):
        pool_size = 2
        stride = 2
        return layers.max_pooling2d(x,
                                    pool_size,
                                    stride,
                                    data_format=Model.DATA_FORMAT,
                                    name='downsampler')

    @staticmethod
    def up_sample(x):
        def wrap_tranpose(f, _x):
            _x = tf.transpose(_x, [0, 2, 3, 1])
            _x = f(_x)
            _x = tf.transpose(_x, [0, 3, 1, 2])

            return _x

        def f(_x):
            _, w, h, _ = _x.shape.as_list()

            _x = tf.image.resize_images(_x,
                                        (w*2, h*2),
                                        method=tf.image.ResizeMethod.BILINEAR)

            return _x

        if Model.DATA_FORMAT=='channels_first':
            return wrap_tranpose(f, x)
        else:
            return f(x)

    @staticmethod
    def _segmap(y):
        y = tf.cast(tf.argmax(y, -1), dtype=tf.uint8)[:,:,:,tf.newaxis]
        return (-10 * y) + 50

    def train(self):
        x, y, w = self.dataset.train
        logits = self.build_graph(x)

        if Model.DATA_FORMAT=='channels_first':
            tf.summary.image('input_image', tf.transpose(x, [0, 2, 3, 1]))
            tf.summary.image('output', Model._segmap(tf.transpose(logits, [0, 2, 3, 1])))
            tf.summary.image('label', Model._segmap(tf.transpose(y, [0, 2, 3, 1])))
        else:
            tf.summary.image('input_image', x)
            tf.summary.image('output', Model._segmap(logits))
            tf.summary.image('label', Model._segmap(y))


        optimize = self.optimizer(self.loss_func(logits, y, w))

        metrics = self.train_metrics(logits, y, w)

        return optimize, metrics

    def test(self):
        x, y, w = self.dataset.test
        logits = self.build_graph(x)

        metrics = self.test_metrics(logits, y, w)

        return logits, metrics

    def train_metrics(self, logits, y, w):
        if self._train_metrics:
            return self._train_metrics(logits, y, w)

        def f(logits, ys):
            return logits, ys

        self._train_metrics = f

        return self._train_metrics(logits, y, w)

    def test_metrics(self, logits, y, w):
        if self._test_metrics:
            return self._test_metrics(logits, y, w)

        def f(logits, y, w):
            return tf.constant(0)

        self._test_metrics = f
        return self._test_metrics(logits, y, w)

    def optimizer(self, loss):
        if self._optimizer:
            return self._optimizer(loss)

        def optimize(loss):
            None

        self._optimizer = optimize

        return self._optimizer(loss)

    def loss_func(self, x, y, w):
        if self._loss_func:
            return self._loss_func(x, y)

        def f(self, x, y, w):
            return tf.nn.softmax_cross_entropy_with_logits_v2(logits=x,
                                                              labels=y)

        self._loss_func = f

        return self._loss_func(x, y)

    def inference(self, x):
        return tf.nn.softmax(self.build_graph(x))


def main():
    info = f"""
    CANDELS Morphological Classification -- Semantic Segmentation
    ConvNet
    """

    tf.logging.set_verbosity(tf.logging.DEBUG)
    print(info)

    in_shape = [5,1,40,40]
    expect_out_shape = [5,9,40,40]

    x = tf.placeholder(tf.float32, shape=in_shape)

    mock_dataset = types.SimpleNamespace(NUM_CLASSES=9)
    Model.DATA_FORMAT = 'channels_first'
    m = Model(mock_dataset, True)

    out = m.build_graph(x)

    assert out.shape.as_list()==expect_out_shape, "Incorrect Shape"

if __name__=='__main__':
    main()