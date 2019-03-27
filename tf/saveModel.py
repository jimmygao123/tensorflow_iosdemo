import gzip
import sys
import struct
import numpy

from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile

train_images_file = "MNIST_data/train-images-idx3-ubyte.gz"
train_labels_file = "MNIST_data/train-labels-idx1-ubyte.gz"
t10k_images_file = "MNIST_data/t10k-images-idx3-ubyte.gz"
t10k_labels_file = "MNIST_data/t10k-labels-idx1-ubyte.gz"


def read32(bytestream):
    # 由于网络数据的编码是大端，所以需要加上>
    dt = numpy.dtype(numpy.int32).newbyteorder('>')
    data = bytestream.read(4)
    return numpy.frombuffer(data, dt)[0]


def read_labels(filename):
    with gzip.open(filename) as bytestream:
        magic = read32(bytestream)
        numberOfLabels = read32(bytestream)
        print(magic)
        print(numberOfLabels)
        labels = numpy.frombuffer(bytestream.read(numberOfLabels), numpy.uint8)
        data = numpy.zeros((numberOfLabels, 10))
        for i in range(len(labels)):
            data[i][labels[i]] = 1
        bytestream.close()
    return data


def read_images(filename):
    # 把文件解压成字节流
    with gzip.open(filename) as bytestream:
        magic = read32(bytestream)
        numberOfImages = read32(bytestream)
        rows = read32(bytestream)
        columns = read32(bytestream)
        images = numpy.frombuffer(bytestream.read(numberOfImages * rows * columns), numpy.uint8)
        images.shape = (numberOfImages, rows * columns)
        images = images.astype(numpy.float32)
        images = numpy.multiply(images, 1.0 / 255.0)
        bytestream.close()
        print(magic)
        print(numberOfImages)
        print(rows)
        print(columns)
    return images


# 解析labels的内容，train_labels包含了60000个数字标签，返回60000个数字标签的数组
train_labels = read_labels(train_labels_file)
# print(labels)
train_images = read_images(train_images_file)

test_labels = read_labels(t10k_labels_file)
# print(labels)
test_images = read_images(t10k_images_file)

import tensorflow as tf

x = tf.placeholder("float", [None, 784.],name='input/x_input')
W = tf.Variable(tf.zeros([784., 10.]))
b = tf.Variable(tf.zeros([10.]))
y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder("float",name='input/y_input')
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
for i in range(1200):
    batch_xs = train_images[50 * i:50 * i + 50]
    batch_ys = train_labels[50 * i:50 * i + 50]
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})


correct_prediction = tf.equal(tf.argmax(y, 1, output_type='int32', name='output'),
                              tf.argmax(y_, 1, output_type='int32'))

# correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print(sess.run(accuracy, feed_dict={x: test_images, y_: test_labels}))

# 保存训练好的模型
# 形参output_node_names用于指定输出的节点名称,output_node_names=['output']对应pre_num=tf.argmax(y,1,name="output"),
output_graph_def = graph_util.convert_variables_to_constants(sess, sess.graph_def, output_node_names=['output'])
with tf.gfile.FastGFile('mnist.pb', mode='wb') as f:  # ’wb’中w代表写文件，b代表将数据以二进制方式写入文件。
    f.write(output_graph_def.SerializeToString())
sess.close()