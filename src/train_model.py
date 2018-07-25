import os

from candels_provider import DataProvider
from unet import Model

import tensorflow as tf

def main():
    tf.logging.set_verbosity(tf.logging.DEBUG)

    params = {
        'data_format':'channels_first',
        'img_files':['hlsp_candels_hst_wfc3_gs-tot_f160w_v1.0_drz.fits'],
        'segmap_file':'segmap.fits',
        'input_size': [40,40],
        'area_size':[80, 80],
        'img_subset':[(16000, 25000), (10500, 20000)],


        'model_dir':'../data/model',


        'train_dir':'../data/report/train/',
        'test_dir':'../data/report/test/',
        'max_iters':10000,
        'batch_size':1,
        'init_learning_rate':1e-6,
        'data_dir':'../data',
        'display_iters':10,
        'test_iters':100,
        'epochs':1,
        'xentropy_coefficient':1,
        'dice_coefficient':1,
        'block_config':[2,2,4,8]
    }

    current_iter = 0
    while (current_iter < params['max_iters']):
        current_iter = train(params)
        test(params)

def fetch_iters():
     with tf.variable_scope("utils", reuse=tf.AUTO_REUSE):
        init = tf.constant_initializer(1, dtype=tf.int32)
        return tf.get_variable('iters',
                            initializer=init,
                            dtype=tf.int32,
                            trainable=False,
                            shape=[])

def eval_metrics(yh, y, weights, metrics_dict):
    """
    yh: network output [n,h,w,c]
    y:  labels         [n,h,w,c]
    """

    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    classes = ['smooth', 'disk', 'start-artifact', 'background']

    with tf.name_scope('metrics'):
        # Cacluate the mean IOU for background/not background at each threshold
        with tf.name_scope('ious'):
            yh_bkg = tf.reshape(tf.nn.sigmoid(yh[:,:,:,-1]), [-1])
            y_bkg = tf.reshape(y[:,:,:,-1], [-1])
            for threshold in thresholds:
                name = 'iou-{}'.format(threshold)
                preds = tf.cast(tf.greater_equal(yh_bkg, threshold), tf.int32)
                metric, update_op = tf.metrics.mean_iou(y_bkg, preds, 2, name=name)
                metrics_dict[name] = update_op
                tf.summary.scalar(name, metric)

        # Calculate the accuracy per class per pixel
        with tf.name_scope('accuracies'):
            y = tf.reshape(y, [-1, 8])[:,:4]
            yh = tf.reshape(yh, [-1, 8])[:,:4]
            lbls = tf.argmax(y, 1)
            preds = tf.argmax(yh, 1)
            for i, c in enumerate(classes):
                in_c = tf.equal(lbls, i)
                metric, update_op = tf.metrics.accuracy(lbls,
                                                        preds,
                                                        weights=in_c,
                                                        name=c)
                metrics_dict[c] = update_op
                tf.summary.scalar(c, metric)

    return metrics_dict


def train(params):
    tf.reset_default_graph()

    DataProvider.NUM_REPEAT = params['epochs']
    dataset = DataProvider(params['img_files'],
                           params['segmap_file'],
                           input_size=params['input_size'],
                           limiting_range=params['img_subset'],
                           batch_size=params['batch_size'])
    iters = fetch_iters()

    learning_rate = tf.train.exponential_decay(params['init_learning_rate'],
                                               iters,
                                               50000,
                                               0.5)

    opt = tf.train.AdamOptimizer(learning_rate)

    # https://arxiv.org/pdf/1701.08816.pdf eq 3, 4, 7, and 8
    def loss_func(logits, y, weights):
        multiclass_loss = tf.nn.softmax_cross_entropy_with_logits_v2
        singleclass_loss = tf.nn.sigmoid_cross_entropy_with_logits

        flat_logits = tf.reshape(logits, [-1, DataProvider.NUM_CLASSES])
        flat_y = tf.reshape(y, [-1, DataProvider.NUM_CLASSES])
        flat_weights = tf.reshape(weights, [-1, 1])

        main_logits, main_y = flat_logits[:,:4], flat_y[:,:4]
        sub_smooth_logits, sub_smooth_y = flat_logits[:,4:7], flat_y[:,4:7]
        sub_disk_logit, sub_disk_y = flat_logits[:,7], flat_y[:,7]

        main_loss = multiclass_loss(logits=main_logits, labels=main_y)
        main_loss = tf.multiply(flat_weights, main_loss)

        sub_smooth_loss = multiclass_loss(logits=sub_smooth_logits,
                                          labels=sub_smooth_y)
        smooth_factor = tf.nn.softmax(sub_smooth_logits)[:,0]
        sub_smooth_loss = tf.multiply(smooth_factor, sub_smooth_loss)
        sub_smooth_loss = tf.multiply(flat_weights, sub_smooth_loss)

        sub_disk_loss = singleclass_loss(logits=sub_disk_logit,
                                         labels=sub_disk_y)
        disk_factor = tf.sigmoid(sub_disk_logit)
        sub_disk_loss = tf.multiply(disk_factor, sub_disk_loss)
        sub_disk_loss = tf.multiply(flat_weights, sub_disk_loss)

        dominant_main_class = tf.argmax(main_y, axis=1, output_type=tf.int32)
        p_dominant_main_class = tf.reduce_max(main_y, axis=1)

        class_coefficient = tf.zeros_like(main_loss)
        for class_idx in range(DataProvider.NUM_CLASSES):
            main_pixels = tf.equal(class_idx, dominant_main_class)
            main_pixels = tf.cast(main_pixels, tf.float32)
            main_pixels = tf.multiply(main_pixels, flat_weights)

            coef = tf.multiply(main_pixels, p_dominant_main_class)
            coef = tf.reduce_mean(coef)
            coef = tf.multiply(coef, main_pixels)
            class_coefficient = tf.add(class_coefficient, coef)

        class_coefficient = tf.divide(1, class_coefficient)
        class_coefficient = 1 / class_coefficient

        classes_loss = tf.add(main_loss, sub_smooth_loss)
        classes_loss = tf.add(classes_loss, sub_disk_loss)

        weighted_xentropy_loss = tf.reduce_mean(classes_loss * class_coefficient)

        # dice loss
        if params['data_format']=='channels_first':
            yh_background = tf.nn.sigmoid(logits[:,4,:,:])
            y_background = y[:,4,:,:]
        else:
            yh_background = tf.nn.sigmoid(logits[:,:,:,4])
            y_background = y[:,:,:,4]

        dice_numerator = tf.reduce_sum(y_background * yh_background,
                                       axis=[1,2])
        dice_denominator = tf.reduce_sum(y_background + yh_background,
                                         axis=[1,2])

        dice_loss = 1 - tf.reduce_mean(2 * dice_numerator / dice_denominator)

        total_loss = params['xentropy_coefficient'] * weighted_xentropy_loss
        total_loss = total_loss + params['dice_coefficient'] * (dice_loss)

        with tf.name_scope('loss'):
            tf.summary.scalar('main_cross_entropy', tf.reduce_mean(main_loss))
            tf.summary.scalar('smooth_loss', tf.reduce_mean(sub_smooth_loss))
            tf.summary.scalar('disk_loss', tf.reduce_mean(sub_disk_loss))
            tf.summary.scalar('weighted_class_loss',
                              tf.reduce_mean(weighted_xentropy_loss))
            tf.summary.scalar('dice_loss', dice_loss)
            tf.summary.scalar('total_loss', total_loss)

        return total_loss

    def optimizer(loss):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            gradients = opt.compute_gradients(loss)
        with tf.name_scope('clipping'):
            clipped = []
            with tf.name_scope('gradients'):
                for g, v in gradients:
                    g = tf.clip_by_value(g, -10, 10)
                    clipped.append((g, v))

        return opt.apply_gradients(gradients, global_step=iters)


    running_metrics = dict()
    def train_metrics(logits, y, weights):
        if params['data_format']=='channels_first':
            _logits = tf.transpose(logits, [0,2,3,1])
            _y = tf.transpose(y, [0,2,3,1])
            eval_metrics(_logits, _y, weights, running_metrics)
        else:
            eval_metrics(logits, y, weights, running_metrics)

        return logits, y


    model = Model(dataset, True)
    model.optimizer = optimizer
    model.train_metrics = train_metrics


    model.loss_func = loss_func

    train = model.train()

    summaries = tf.summary.merge_all()

    metric_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope='metrics/*')
    metric_reset = tf.variables_initializer(metric_vars)

    # start training
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        if len(os.listdir(params['model_dir'])) > 0:
            latest_checkpoint = tf.train.latest_checkpoint(params['model_dir'])
            saver.restore(sess, latest_checkpoint)
        else:
            sess.run(init)

        writer = tf.summary.FileWriter(params['train_dir'], graph=sess.graph)

        for _ in range(params['test_iters']):
            current_iter = iters.eval()

            try:
                if current_iter % 10 == 0:
                    sess.run(metric_reset)
                    run_ops = [train] + list(running_metrics.values())
                    sess.run(run_ops)
                    s = sess.run(summaries)
                    writer.add_summary(s, current_iter)
                    save_name = '{}.ckpt'.format(Model.NAME)
                    saver.save(sess,
                               os.path.join(params['model_dir'], save_name),
                               global_step=iters)
                else:
                    sess.run([train])
            except tf.errors.OutOfRangeError:
                saver.save(sess, params['model_dir'], global_step=iters)
                break

        writer.flush()
        writer.close()
        return current_iter



def test(params):
    tf.reset_default_graph()
    DataProvider.NUM_REPEAT = params['epochs']
    dataset = DataProvider(params['img_files'],
                           params['segmap_file'],
                           input_size=params['input_size'],
                           limiting_range=params['img_subset'],
                           batch_size=params['batch_size'])
    iters = fetch_iters()

    running_metrics = dict()
    def test_metrics(logits, y, weights):
        if params['data_format']=='channels_first':
            _logits = tf.transpose(logits, [0,2,3,1])
            _y = tf.transpose(y, [0,2,3,1])
            eval_metrics(_logits, _y, weights, running_metrics)
        else:
            eval_metrics(logits, y, weights, running_metrics)

        return logits, y

    model = Model(dataset, False)
    model.test_metrics = test_metrics

    test, _ = model.test()
    summaries = tf.summary.merge_all()


    metric_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope='metrics/*')
    metric_reset = tf.variables_initializer(metric_vars)

    # restore graph
    saver = tf.train.Saver()
    with tf.Session() as sess:
        latest_checkpoint = tf.train.latest_checkpoint(params['model_dir'])
        saver.restore(sess, latest_checkpoint)
        sess.run(metric_reset)

        writer = tf.summary.FileWriter(params['test_dir'], graph=sess.graph)

        # go through the whole test set
        try:
            run_ops = [test] + list(running_metrics.values())
            while True:
                sess.run([run_ops])
        except tf.errors.OutOfRangeError:
            pass

        s = sess.run(summaries)
        writer.add_summary(s, iters.eval())
        writer.flush()
        writer.close()

if __name__=='__main__':
    main()