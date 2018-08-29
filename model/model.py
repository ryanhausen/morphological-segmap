import tensorflow as tf

from ryanml.models import unet

class Model(unet.Model):
    """Model class for {paper name}

    Args:
        hparams (tf.train.HParams): Model Hyperparameters
        dataset (object): dataset to use for training
        data_format: channels_first or channels_last

    Required HParams:
        inference (bool): true if using pretrained model
        down_filters (list): number of filters for each down conv section
        num_down_convs (int): number of conv ops per down conv section
        up_filters (list): number of filters for each up conv section
        num_up_convs (int): number of conv ops per up conv section
        batch_norm (bool): use batch normalization
        dropout (bool): use dropout

    Optional HParams:
        learning_rate (float): learning rate for training, required if inference
                               is set to false
        dropout_rate (float): the percentage of neurons to drop [0.0, 1.0]


    Note if you are using a pretrained weights for inference only you need
    to mock the dataset object. You can do so with the following code:
    ```
    from collections import namedtuple

    Dataset = namedtuple('Dataset', ['num_labels'])
    dataset = Dataset(5)
    ```
    """

    def __init__(self, hparams, dataset, data_format):
        super().__init__(hparams, dataset, data_format)
        if not hparams.inference:
            self.opt = tf.train.AdamOptimizer(hparams.learning_rate)


    def loss_func(self, logits, labels):
        # softmax loss, per pixel
        flat_logits = tf.reshape(logits, [-1, 5])
        flat_y = tf.reshape(labels, [-1, 5])
        xentropy_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=flat_logits,
                                                                   labels=flat_y)

        # class coeffecients, different from the paper above we don't have
        # one-hot classes per pixel and so instead of a hard count, we'll
        # used an expected pixel account
        dominant_class = tf.argmax(flat_y, axis=1, output_type=tf.int32)
        p_dominant_class = tf.reduce_max(flat_y, axis=1)

        class_coefficient = tf.zeros_like(xentropy_loss)
        for output_class_idx in range(5):
            class_pixels = tf.cast(tf.equal(output_class_idx, dominant_class),
                                   tf.float32)

            coef = tf.reduce_mean(class_pixels * p_dominant_class)
            class_coefficient = tf.add(class_coefficient, coef*class_pixels)

        class_coefficient = 1 / class_coefficient

        weighted_xentropy_loss = tf.reduce_mean(xentropy_loss * class_coefficient)

        # dice loss
        if self.data_format=='channels_first':
            yh_background = tf.nn.sigmoid(logits[:,-1,:,:])
            y_background = labels[:,-1,:,:]
        else:
            yh_background = tf.nn.sigmoid(logits[:,:,:,-1])
            y_background = labels[:,:,:,-1]

        dice_numerator = tf.reduce_sum(y_background * yh_background,
                                       axis=[1,2])
        dice_denominator = tf.reduce_sum(y_background + yh_background,
                                         axis=[1,2])

        dice_loss = tf.reduce_mean(2 * dice_numerator / dice_denominator)

        total_loss = weighted_xentropy_loss
        total_loss = total_loss + (1-dice_loss)

        return total_loss

    def optimizer(self, loss):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optimize = self.opt.minimize(loss)

        return optimize

    @staticmethod
    def eval_metrics(yh, y):
        """
        yh: network output [n,h,w,c]
        y:  labels         [n,h,w,c]
        """
        metrics_dict = {}

        thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
        classes = ['spheroid', 'disk', 'irregular', 'point_source', 'background']

        yh_bkg = tf.reshape(tf.nn.sigmoid(yh[:,:,:,-1]), [-1])
        y_bkg = tf.reshape(y[:,:,:,-1], [-1])
        for threshold in thresholds:
            name = 'iou-{}'.format(threshold)
            with tf.name_scope(name):
                preds = tf.cast(tf.greater_equal(yh_bkg, threshold), tf.int32)
                metric, update_op = tf.metrics.mean_iou(y_bkg, preds, 2, name=name)
                metrics_dict[name] = (metric, update_op)

        # Calculate the accuracy per class per pixel
        y = tf.reshape(y, [-1, 5])
        yh = tf.reshape(yh, [-1, 5])
        lbls = tf.argmax(y, 1)
        preds = tf.argmax(yh, 1)

        name = 'overall'
        metric, update_op = tf.metrics.accuracy(lbls,
                                                preds,
                                                name=name)

        metrics_dict[name] = (metric, update_op)
        for i, c in enumerate(classes):
            in_c = tf.equal(lbls, i)
            name = classes[i]
            metric, update_op = tf.metrics.accuracy(lbls,
                                                    preds,
                                                    weights=in_c,
                                                    name=name)
            metrics_dict[name] = (metric, update_op)

        return metrics_dict


    def train_metrics(self, logits, labels):
        with tf.name_scope('train_metrics'):
            metrics_dict = Model.eval_metrics(logits, labels)

        names, finalize, running = [], [], []

        for key in sorted(metrics_dict):
            names.append(key)
            finalize.append(metrics_dict[key][0])
            running.append(metrics_dict[key][1])

        return ([names, finalize], running)


    def test_metrics(self, logits, labels):
        with tf.name_scope('test_metrics'):
            metrics_dict = Model.eval_metrics(logits, labels)

        names, finalize, running = [], [], []

        for key in sorted(metrics_dict):
            names.append(key)
            finalize.append(metrics_dict[key][0])
            running.append(metrics_dict[key][1])

        return ([names, finalize], running)

    def inference(self, x):
        return tf.nn.softmax(self.build_graph(x, False))
