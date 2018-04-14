import csv

import numpy as np
import tensorflow as tf

from text_encoder import *
from stackoverflow import StackOverflowScraper


line_length = 80

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

epochs = 10
with tf.Session() as sess:
    print('building model')
    # input 1 line of line_length chars of one of 96 classes
    X = tf.placeholder(tf.float32, [line_length, 96])
    # outputs classified line
    Y_ = tf.placeholder(tf.float32, [1])
    W0 = tf.Variable(tf.truncated_normal([line_length * 96, 80]))
    W1 = tf.Variable(tf.truncated_normal([80, 1]))
    # b0 = tf.Variable(tf.truncated_normal([line_length, 96]))
    # Y = tf.nn.sigmoid(tf.matmul(X, W0) + b0)
    XX = tf.reshape(X, [-1, line_length * 96])
    H1 = tf.nn.sigmoid(tf.matmul(XX, W0))
    Y = tf.nn.sigmoid(tf.matmul(H1, W1))

    loss_func = abs(Y_ - Y)
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss_func)

    sess.run(tf.global_variables_initializer())

    print('pre-train benchmark: ', end='')
    accuracies = []
    for i, line in enumerate(test_lines):
        label = np.array(test_labels[i])
        prediction = sess.run(Y, feed_dict={X: line, Y_: label})
        accuracy = (test_labels[i] == prediction.round()).all(axis=-1)
        accuracies.append(accuracy)
    print(np.mean(accuracies))

    print('training...')
    for epoch in range(epochs):
        losses = []
        for i, line in enumerate(lines):
            label = np.array(labels[i])
            _, loss, prediction = sess.run(
                    [train_step, loss_func, Y], feed_dict={X: line, Y_: label})
            losses.append(loss)
        if (not epoch) or (not (epoch + 1) % 1):  # mod value sets epochs per print
            print('\tloss: ', np.mean(losses))

    print('trained perfromance: ', end='')
    accuracies = []
    predictions= []
    for i, line in enumerate(test_lines):
        label = np.array(test_labels[i])
        prediction = sess.run(Y, feed_dict={X: line, Y_: label})
        # accuracy = (test_labels[i] == prediction.round()).all(axis=-1)
        accuracy = test_labels[i] == prediction.round()
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
