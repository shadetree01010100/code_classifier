import csv
import os
import numpy as np
import tensorflow as tf
from text_encoder import *


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # supress TF build warnings

# INPUT OPTIONS
line_length = 80
char_classes = 96

# RUN OPTIONS
epochs = 10
print_interval = 1

# NN OPTIONS
outputs = 1
H0_count = 80

lines = []
labels = []
test_lines = []
test_labels = []

with open('train_stackoverflow.csv', newline='', encoding='utf-8') as csvfile:
    train_data = csv.reader(csvfile)
    for row in train_data:
        labels.append([int(row[0])])
        lines.append(encode_line(row[1]))
with open('test_stackoverflow.csv', newline='', encoding='utf-8') as csvfile:
    test_data = csv.reader(csvfile)
    for row in test_data:
        test_labels.append([int(row[0])])
        test_lines.append(encode_line(row[1]))

lines = np.array(lines)
labels = np.array(labels)
test_lines = np.array(test_lines)
lest_labels = np.array(test_labels)
with tf.Session() as sess:
    print('building model')
    # tf.set_random_seed(1)
    # input, 1 line of line_length chars of one of char_classes
    X = tf.placeholder(tf.float32, [None, line_length, char_classes])
    # desired output, known label for that line
    Y_ = tf.placeholder(tf.float32, [None, outputs])

    flattened_length = line_length * char_classes
    XX = tf.reshape(X, [-1, flattened_length]) #  flatten for dense layers
    W0 = tf.Variable(tf.truncated_normal([flattened_length, H0_count]))
    B0 = tf.Variable(tf.truncated_normal([H0_count]))
    W1 = tf.Variable(tf.truncated_normal([H0_count, outputs]))
    B1 = tf.Variable(tf.truncated_normal([outputs]))

    H0 = tf.nn.sigmoid(tf.matmul(XX, W0) + B0)
    Y = tf.nn.sigmoid(tf.matmul(H0, W1) + B1)

    loss_func = abs(Y_ - Y)
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss_func)

    sess.run(tf.global_variables_initializer())

    print('pre-train benchmark: ', end='')
    accuracies = []
    for i, line in enumerate(test_lines):
        label = test_labels[i]
        _batch = np.expand_dims(line, axis=0)
        _label = np.expand_dims(label, axis=0)
        prediction = sess.run(Y, feed_dict={X: _batch, Y_: _label})
        accuracy = (test_labels[i] == prediction.round()).all(axis=-1)
        accuracies.append(accuracy)
    print(np.mean(accuracies))

    print('training...')
    for epoch in range(epochs):
        losses = []
        for i, line in enumerate(lines):
            label = labels[i]
            _batch = np.expand_dims(line, axis=0)
            _label = np.expand_dims(label, axis=0)
            _, loss, prediction = sess.run(
                    [train_step, loss_func, Y], feed_dict={X: _batch, Y_: _label})
            losses.append(loss)
        if (not epoch) or (not (epoch + 1) % print_interval):
            print('\tloss: ', np.mean(losses))

    print('trained perfromance: ', end='')
    accuracies = []
    predictions= []
    for i, line in enumerate(test_lines):
        label = test_labels[i]
        _batch = np.expand_dims(line, axis=0)
        _label = np.expand_dims(label, axis=0)
        prediction = sess.run(Y, feed_dict={X: _batch, Y_: _label})
        accuracy = (test_labels[i] == prediction.round()).all(axis=-1)
        # accuracy = test_labels[i] == prediction.round()
        accuracies.append(accuracy)
        predictions.append(prediction)
    print(np.mean(accuracies))

for i, line in enumerate(test_lines):
    input('press [enter] to view test posts')
    print(
        decode_line(line),
        '\n',
        test_labels[i],
        [round(predictions[i][0][0] * 100, 1)])
