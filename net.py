import csv
import os
import time
import numpy as np
import tensorflow as tf

from text_encoder import *


def _open(filename):
    """ Helper method to open(filename) with appropriate args."""
    return open(filename, newline='', encoding='utf-8')

def _print(line='', newline=True, overwrite=False):
    """ Helper method to print and update lines."""
    end = '\n' if newline else ''
    start = '\r' if overwrite else ''
    print('{}{}'.format(start, line), end=end, flush=True)

# supress TF build info logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Force compatability of convolution on RTX series GPU
# https://github.com/tensorflow/tensorflow/issues/24496
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

# CNN CONFIG
# INPUT OPTIONS
line_length = LINE_LENGTH
char_classes = NUM_CHARS

# RUN OPTIONS
epochs = 20
learning_rate = 0.001

# CNN LAYERS
# TODO: moar
C0_depth = 16
C0_width = 4
C0_stride = 1

# NN LAYERS
H0_count = 80
outputs = 1

session = tf.Session(config=config)
with session:
    ### debug ###
    _print('building model ', newline=False)
    start_time = time.monotonic()
    ### debug ###

    # tf.set_random_seed(1)
    # input, 1 line of line_length chars of one of char_classes
    X = tf.placeholder(tf.float32, [None, line_length, char_classes])
    # desired output, known label for that line
    Y_ = tf.placeholder(tf.float32, [None, outputs])

    # convolutional layers
    CW0 = tf.Variable(tf.ones([C0_width, char_classes, C0_depth]) / 10)
    CB0 = tf.Variable(tf.ones([C0_depth]) / 10)
    C0 = tf.nn.relu(tf.nn.conv1d(X, CW0, C0_stride, 'SAME') + CB0)

    # flatten for input to neural network layers
    flattened_length = C0_depth * H0_count
    XX = tf.reshape(C0, [-1, flattened_length])

    # hidden layers
    W0 = tf.Variable(tf.truncated_normal([flattened_length, H0_count]))
    B0 = tf.Variable(tf.truncated_normal([H0_count]))
    H0 = tf.nn.sigmoid(tf.matmul(XX, W0) + B0)
    # output layer
    W1 = tf.Variable(tf.truncated_normal([H0_count, outputs]))
    B1 = tf.Variable(tf.truncated_normal([outputs]))
    Y = tf.nn.sigmoid(tf.matmul(H0, W1) + B1)

    # setup training
    loss_func = tf.losses.absolute_difference(Y_, Y)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_step = optimizer.minimize(loss_func)

    ### debug ###
    elapsed = round(time.monotonic() - start_time, 4)
    _print('({})'.format(elapsed))
    _print('setting up tensorboard ', newline=False)
    start_time = time.monotonic()
    ### debug ###

    summary_writer = tf.summary.FileWriter(
        './train/{}'.format(int(time.time())),
        session.graph,
    )

    ### debug ###
    elapsed = round(time.monotonic() - start_time, 4)
    _print('({})'.format(elapsed))
    _print('initializing variables ', newline=False)
    start_time = time.monotonic()
    ### debug ###

    session.run(tf.global_variables_initializer())

    ### debug ###
    elapsed = round(time.monotonic() - start_time, 4)
    print('({})'.format(elapsed))
    _print()
    ### debug ###

    accuracies = []
    batches = 0

    ### debug ###
    _print('opening test data ', newline=False)
    start_time = time.monotonic()
    ### debug ###

    with _open('test_stackoverflow.csv') as csvfile:
        test_data = csv.reader(csvfile)
        elapsed = round(time.monotonic() - start_time, 4)

        ### debug ###
        print('({})'.format(elapsed))
        _print('benchmarking ', newline=False)
        start_time = time.monotonic()
        ### debug ###

        for row in test_data:
            try:
                # batch size = 1
                _batch = np.expand_dims(encode_line(row[1]), axis=0)
                _label = np.expand_dims([int(row[0])], axis=0)
            except IndexError:
                # empty line at end of post
                continue
            prediction = session.run(Y, feed_dict={X: _batch, Y_: _label})
            accuracy = ([int(row[0])] == prediction.round()).all(axis=-1)
            accuracies.append(accuracy)
            batches += 1

    ### debug ###
    elapsed = round(time.monotonic() - start_time, 4)
    _print(
        '({} ({} batches: {} avg.))'.format(
            elapsed,
            batches,
            round(elapsed / batches, 4),
        ),
    )
    _print(
        'pretrain benchmark: {} % correct'.format(
            round(np.mean(accuracies) * 100, 3),
        ),
    )
    _print()
    ### debug ###

    global_step = 0
    losses = []
    for epoch in range(epochs):
        batches = 0

        ### debug ###
        if not epoch:
            _print('opening train data ', newline=False)
            start_time = time.monotonic()
        ### debug ###

        with _open('train_stackoverflow.csv') as csvfile:
            train_data = csv.reader(csvfile)

            ### debug ###
            if not epoch:
                elapsed = round(time.monotonic() - start_time, 4)
                _print('({})'.format(elapsed))
                _print(
                    'training epoch {}/{} '.format(epoch + 1, epochs),
                    newline=False,
                )
            start_time = time.monotonic()
            ### debug ###

            lines = []
            labels = []
            for row in train_data:
                try:
                    lines.append(encode_line(row[1]))
                    labels.append([int(row[0])])
                except IndexError:
                    # empty line at end of post, run the batch!
                    _, loss, prediction = session.run(
                        [train_step, loss_func, Y],
                        feed_dict={X: np.array(lines), Y_: np.array(labels)})
                    losses.append(loss)
                    # prepare for next batch
                    lines = []
                    labels = []
                    batches += 1
        average_loss = np.mean(losses)
        summary = tf.Summary()
        summary.value.add(tag='average loss', simple_value=average_loss)
        summary_writer.add_summary(summary, epoch + 1)
        losses = []

        ### debug ###
        elapsed = round(time.monotonic() - start_time, 4)
        _print(
            'training epoch {}/{} ({}, {} avg. per batch)     '.format(
                epoch + 1,
                epochs,
                elapsed,
                round(elapsed / batches, 4),
            ),
            newline=False,
            overwrite=True,
        )
        ### debug ###

    summary_writer.close()

    _print()
    _print('trained performance: ', newline=False)
    accuracies = []
    with _open('test_stackoverflow.csv') as csvfile:
        test_data = csv.reader(csvfile)
        for row in test_data:
            try:
                # batch size = 1
                _batch = np.expand_dims(encode_line(row[1]), axis=0)
                _label = np.expand_dims([int(row[0])], axis=0)
            except IndexError:
                # empty line at end of post
                continue
            prediction = session.run(Y, feed_dict={X: _batch, Y_: _label})
            accuracy = ([int(row[0])] == prediction.round()).all(axis=-1)
            accuracies.append(accuracy)
            batches += 1
    _print('{} % correct'.format(round(np.mean(accuracies) * 100, 3)))
    summary_writer.close()
