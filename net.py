import csv
import os
import time
import numpy as np
import tensorflow as tf

from text_encoder import *


def _open(filename):
    """ Helper method to open(filename) with appropriate args."""
    return open(filename, newline='', encoding='utf-8')

# supress TF build info logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Force compatability of convolution on RTX series GPU
# https://github.com/tensorflow/tensorflow/issues/24496
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

# CNN CONFIG
# INPUT OPTIONS
line_length = 80
char_classes = 96

# RUN OPTIONS
epochs = 20
print_interval = 1

# CNN LAYERS
C0_depth = 4
C0_width = 3
C0_stride = 1

# NN LAYERS
H0_count = 80
outputs = 1

with tf.Session(config=config) as sess:
    print('building model', end='')
    start_time = time.monotonic()
    # tf.set_random_seed(1)
    # input, 1 line of line_length chars of one of char_classes
    X = tf.placeholder(tf.float32, [None, line_length, char_classes])
    # desired output, known label for that line
    Y_ = tf.placeholder(tf.float32, [None, outputs])

    # convolutional layers
    CW0 = tf.Variable(tf.truncated_normal([C0_width, char_classes, C0_depth]))
    CB0 = tf.Variable(tf.truncated_normal([C0_depth]))
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

    loss_func = abs(Y_ - Y)
    train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss_func)
    elapsed = round(time.monotonic() - start_time, 4)
    print(' ({})'.format(elapsed))

    print('setting up tensorboard', end='')
    start_time = time.monotonic()
    summary_writer = tf.summary.FileWriter(
        './train/{}'.format(int(time.time())),
        sess.graph,
    )
    elapsed = round(time.monotonic() - start_time, 4)
    print(' ({})'.format(elapsed))

    print('initializing variables', end='')
    start_time = time.monotonic()
    sess.run(tf.global_variables_initializer())
    elapsed = round(time.monotonic() - start_time, 4)
    print(' ({})'.format(elapsed))

    print()
    accuracies = []
    trials = 0
    print('opening test data', end='')
    start_time = time.monotonic()
    with _open('test_stackoverflow.csv') as csvfile:
        test_data = csv.reader(csvfile)
        elapsed = round(time.monotonic() - start_time, 4)
        print(' ({})'.format(elapsed))
        print('benchmarking', end='\r')
        start_time = time.monotonic()
        for row in test_data:
            try:
                # batch size = 1
                _batch = np.expand_dims(encode_line(row[1]), axis=0)
                _label = np.expand_dims([int(row[0])], axis=0)
            except IndexError:
                # empty line indicates end of post
                continue
            prediction = sess.run(Y, feed_dict={X: _batch, Y_: _label})
            accuracy = ([int(row[0])] == prediction.round()).all(axis=-1)
            accuracies.append(accuracy)
            trials += 1
    elapsed = round(time.monotonic() - start_time, 4)
    print('benchmarking ({} ({} trials: {} avg.))'.format(
        elapsed,
        trials,
        round(elapsed / trials, 4)))
    print('pretrain benchmark: {}%'.format(
        round(np.mean(accuracies) * 100, 3)))

    print()
    trials = 0
    for epoch in range(epochs):
        losses = []
        if not epoch:
            print('opening train data', end='')
            start_time = time.monotonic()
        with _open('train_stackoverflow.csv') as csvfile:
            train_data = csv.reader(csvfile)
            if not epoch:
                elapsed = round(time.monotonic() - start_time, 4)
                print(' ({})'.format(elapsed))
                print('training', end='\r')
            start_time = time.monotonic()
            for row in train_data:
                try:
                    # batch size = 1
                    _batch = np.expand_dims(encode_line(row[1]), axis=0)
                    _label = np.expand_dims([int(row[0])], axis=0)
                except IndexError:
                    # empty line, end of post
                    continue
                _, loss, prediction = sess.run(
                        [train_step, loss_func, Y],
                        feed_dict={X: _batch, Y_: _label})
                losses.append(loss)
                trials += 1
        elapsed = round(time.monotonic() - start_time, 4)
        print(
            'training ({} (epoch {}/{}, {} trials: {} avg.))    '.format(
                elapsed,
                epoch + 1,
                epochs,
                trials,
                round(elapsed / trials, 4),
            end='\r',
            flush=True,
        ))

    print()
    print('trained perfromance: ', end='')
    accuracies = []
    predictions= []
    with _open('test_stackoverflow.csv') as csvfile:
        test_data = csv.reader(csvfile)
        for row in test_data:
            try:
                # batch size = 1
                _batch = np.expand_dims(encode_line(row[1]), axis=0)
                _label = np.expand_dims([int(row[0])], axis=0)
            except IndexError:
                # empty line indicates end of post
                continue
            prediction = sess.run(Y, feed_dict={X: _batch, Y_: _label})
            accuracy = ([int(row[0])] == prediction.round()).all(axis=-1)
            accuracies.append(accuracy)
            predictions.append(prediction)
    print(round(np.mean(accuracies) * 100, 3))
    summary_writer.close()

with _open('test_stackoverflow.csv') as csvfile:
    test_data = csv.reader(csvfile)
    for i, row in enumerate(test_data):
        try:
            input('press [enter] to view test posts')
            print(
                decode_line(row[1]),
                '\n',
                bool(int(row[0])),
                [round(predictions[i][0][0] * 100, 1)])
        except IndexError:
            # empty line indicates end of post
            continue
