import tensorflow as tf
from text_encoder import *
def encode_line(label, text):
    temp = [encode_char(t - 32) for t in text[:LINE_LENGTH]]
    for _ in range(LINE_LENGTH - len(temp)):
            temp.append(encode_char(NUM_CHARS - 1))
    return int(label), temp

sess=tf.Session()
filenames=['test_stackoverflow.csv', 'train_stackoverflow.csv']
record_defaults=[tf.string, tf.string]
dataset=tf.data.experimental.CsvDataset(filenames, record_defaults)
dataset=dataset.map(
    lambda label, line: tuple(tf.py_func(
            encode_line, [label, line], [int64, tf.uint8])))
dataset=dataset.batch(1)
handle = tf.placeholder(tf.string, shape=[])
iterator = tf.data.Iterator.from_string_handle(
    handle, dataset.output_types, dataset.output_shapes)
next_element = iterator.get_next()
training_iterator = dataset.make_one_shot_iterator()
training_handle = sess.run(training_iterator.string_handle())

sess.run(next_element, feed_dict={handle: training_handle})
# (array([b'0'], dtype=object), array([b'What are metaclasses and what do we use them for?'], dtype=object))
