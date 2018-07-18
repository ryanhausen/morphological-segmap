"""
Base class for models
"""
import tensorflow as tf
from tf_src import tf_logger as log

class Model:
    NAME = "base_model"
    DATA_FORMAT = "channels_first"

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


    def model_fn(self, x):
        raise NotImplementedError()

    def build_graph(self, x):
        if self._graph:
            return self._graph(x)

        self._graph = self.model_fn
        return self._graph(x)

    def train(self):
        x, y = self.dataset.train
        logits = self.build_graph(x)
        optimize = self.optimizer(self.loss_func(logits, y))
        metrics = self.train_metrics(logits, y)

        return optimize, metrics

    def test(self):
        x, y = self.dataset.test
        logits = self.build_graph(x)
        metrics = self.test_metrics(logits, y)

        return logits, metrics

    def train_metrics(self, logits, y):
        if self._train_metrics:
            return self._train_metrics(logits, y)

        def f(logits, ys):
            log.warn('No training metrics set')
            return logits, ys

        self._train_metrics = f

        return self._train_metrics(logits, y)

    def test_metrics(self, logits, y):
        if self._test_metrics:
            return self._test_metrics(logits, y)

        def f(logits, y):
            log.warn('No testing metrics set')
            return logits, y

        self._test_metrics = f
        return self._test_metrics(logits, y)

    def optimizer(self, loss):
        if self._optimizer:
            return self._optimizer(loss)

        def optimize(loss):
            log.warn('No optimizer set')

        self._optimizer = optimize

        return self._optimizer(loss)

    def loss_func(self, x, y):
        if self._loss_func:
            return self._loss_func(x, y)

        def f(self, x, y):
            log.warn('Using default loss cross entropy')
            return tf.nn.softmax_cross_entropy_with_logits_v2(logits=x,
                                                              labels=y)
        tf.losses.add_loss(f)

        self._loss_func = f


    def inference(self, x):
        return tf.nn.softmax(self.build_graph(x))