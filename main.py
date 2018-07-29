#!/usr/bin/env python3
from distutils.version import LooseVersion
import logging
import os.path

import tensorflow as tf

import helper
import project_tests as tests


logger = logging.getLogger(__name__)


# Check TensorFlow Version
assert (LooseVersion(tf.__version__) >= LooseVersion('1.0'),
        'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__))
logger.info('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    logger.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Loads pre-trained VGG Model into TensorFlow.

    :param sess: TF session.
    :param vgg_path: Path to VGG folder, containing "variables/" and "saved_model.pb".
    :return: Tuple of Tensors from VGG model:
             (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    vgg_tag = 'vgg16'
    tensor_names = ['image_input:0',
                    'keep_prob:0',
                    'layer3_out:0',
                    'layer4_out:0',
                    'layer7_out:0']

    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    graph = tf.get_default_graph()
    return [graph.get_tensor_by_name(n) for n in tensor_names]


tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Creates the layers for a fully convolutional network.
    Builds skip-layers using the VGG layers.

    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # Freeze the VGG network.
    vgg_layer7_out = tf.stop_gradient(vgg_layer7_out)
    vgg_layer4_out = tf.stop_gradient(vgg_layer4_out)
    vgg_layer3_out = tf.stop_gradient(vgg_layer3_out)

    # Batch normalize the VGG layers.
    vgg_layer7_out = tf.layers.batch_normalization(vgg_layer7_out)
    vgg_layer4_out = tf.layers.batch_normalization(vgg_layer4_out)
    vgg_layer3_out = tf.layers.batch_normalization(vgg_layer3_out)

    # Scale layer 3 and 4.
    vgg_layer4_out = tf.multiply(vgg_layer4_out, 0.01)
    vgg_layer3_out = tf.multiply(vgg_layer3_out, 0.0001)

    # Layer 7: 1 by 1 convolution.
    layer7_1x1 = tf.layers.conv2d(
        vgg_layer7_out, num_classes, 1, 1,
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
        activation=tf.nn.relu)

    # Layer 7: upsample and batch normalize.
    layer7_upsampled = tf.layers.conv2d_transpose(
        layer7_1x1, num_classes, 4, 2, padding='same',
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    layer7_upsampled = tf.layers.batch_normalization(layer7_upsampled)

    # Layer 4: 1 by 1 convolution and batch normalize.
    layer4_1x1 = tf.layers.conv2d(
        vgg_layer4_out, num_classes, 1, 1,
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    layer4_1x1 = tf.layers.batch_normalization(layer4_1x1)

    # Layer 4_7: combine.
    layer_4_7 = tf.add(layer7_upsampled, layer4_1x1)

    # Layer 4_7: upsample and batch normalize.
    layer_4_7_upsampled = tf.layers.conv2d_transpose(
        layer_4_7, num_classes, 4, 2, padding='same',
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    layer_4_7_upsampled = tf.layers.batch_normalization(layer_4_7_upsampled)

    # Layer 3: 1 by 1 convolution and batch normalize.
    layer3_1x1 = tf.layers.conv2d(
        vgg_layer3_out, num_classes, 1, 1,
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),)
    layer3_1x1 = tf.layers.batch_normalization(layer3_1x1)

    # Layer 3_4_7: combine.
    layer_3_4_7 = tf.add(layer_4_7_upsampled, layer3_1x1)

    # Layer 3_4_7: sample.
    return tf.layers.conv2d_transpose(
        layer_3_4_7, num_classes, 16, 8, padding='same',
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))


tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Builds the TensorFLow loss and optimizer operations.

    :param nn_last_layer: TF Tensor of the last layer in the neural network.
    :param correct_label: TF Placeholder for the correct label image.
    :param learning_rate: TF Placeholder for the learning rate.
    :param num_classes: Number of classes to classify.
    :return: Tuple of (logits, train_op, cross_entropy_loss).
    """
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    correct_label = tf.reshape(correct_label, (-1, num_classes))

    is_correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(correct_label, 1))
    accuracy_op = tf.reduce_mean(tf.cast(is_correct_prediction, tf.float32))

    cross_entropy_loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label))
    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy_loss)

    return logits, train_op, cross_entropy_loss, accuracy_op


tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Trains neural network and print out the loss during training.

    :param sess: TF Session.
    :param epochs: Number of epochs.
    :param batch_size: Batch size.
    :param get_batches_fn: Function to get batches of training data.
                           Call using get_batches_fn(batch_size).
    :param train_op: TF Operation to train the neural network.
    :param cross_entropy_loss: TF Tensor for the amount of loss.
    :param input_image: TF Placeholder for input images.
    :param correct_label: TF Placeholder for label images.
    :param keep_prob: TF Placeholder for dropout keep probability.
    :param learning_rate: TF Placeholder for learning rate.
    """
    for epoch in range(epochs):
        for i, (images, labels) in enumerate(get_batches_fn(batch_size)):
            loss, _ = sess.run([cross_entropy_loss, train_op], feed_dict={
                input_image: images,
                correct_label: labels,
                keep_prob: 0.8,
                learning_rate: 1e-4
            })
            logger.info('epoch {}, batch {}, loss={}'.format(epoch, i, loss))


tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pre-trained VGG model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        learning_rate = tf.placeholder(tf.float32)
        correct_label = tf.placeholder(tf.int32, [None, None, None, num_classes])

        # Build NN using load_vgg, layers, and optimize function.
        input_image, keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out = load_vgg(sess, vgg_path)
        layers_output = layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes)
        logits, train_op, cross_entropy_loss, _ = optimize(
            layers_output, correct_label, learning_rate, num_classes)

        # Train NN using the train_nn function.
        logger.info('Starting training')
        sess.run(tf.global_variables_initializer())
        train_nn(sess, epochs=20, batch_size=8,
                 keep_prob=keep_prob,
                 get_batches_fn=get_batches_fn,
                 train_op=train_op,
                 cross_entropy_loss=cross_entropy_loss,
                 input_image=input_image,
                 correct_label=correct_label,
                 learning_rate=learning_rate)
        logger.info('Done training')

        # Save inference data using helper.save_inference_samples.
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, filename='training.log')
    run()
