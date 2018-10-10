# An undercomplete autoencoder on MNIST dataset
from __future__ import division, print_function, absolute_import
import tensorflow.contrib.layers as lays

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from skimage import transform
from tensorflow.examples.tutorials.mnist import input_data


def encode(inputs):
    ### Encoder
    conv1 = tf.layers.conv2d(inputs=inputs, filters=32, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
    # Now 28x28x32
    maxpool1 = tf.layers.max_pooling2d(conv1, pool_size=(2,2), strides=(2,2), padding='same')
    # Now 14x14x32
    conv2 = tf.layers.conv2d(inputs=maxpool1, filters=32, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
    # Now 14x14x32
    maxpool2 = tf.layers.max_pooling2d(conv2, pool_size=(2,2), strides=(2,2), padding='same')
    # Now 7x7x32
    conv3 = tf.layers.conv2d(inputs=maxpool2, filters=16, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
    # Now 7x7x16
    encoded = tf.layers.max_pooling2d(conv3, pool_size=(2,2), strides=(2,2), padding='same')
    # Now 4x4x16

    return encoded

def decode(inputs):
    ### Decoder
    upsample1 = tf.image.resize_images(inputs, size=(7,7), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    # Now 7x7x16
    conv4 = tf.layers.conv2d(inputs=upsample1, filters=16, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
    # Now 7x7x16
    upsample2 = tf.image.resize_images(conv4, size=(14,14), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    # Now 14x14x16
    conv5 = tf.layers.conv2d(inputs=upsample2, filters=32, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
    # Now 14x14x32
    upsample3 = tf.image.resize_images(conv5, size=(28,28), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    # Now 28x28x32
    conv6 = tf.layers.conv2d(inputs=upsample3, filters=32, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
    # Now 28x28x32

    logits = tf.layers.conv2d(inputs=conv6, filters=1, kernel_size=(3,3), padding='same', activation=None)
    #Now 28x28x1

    # Pass logits through sigmoid to get reconstructed image
    decoded = tf.nn.sigmoid(logits)

    return decoded



def conv_autoencoder(inputs, targets, learning_rate):
    enc_net = encode(inputs)
    dec_net = decode(enc_net)
    # Pass decoded image through sigmoid and calculate the cross-entropy loss
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=targets, logits=dec_net)
    # Get cost and define the optimizer
    cost = tf.reduce_mean(loss)
    opt = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    return enc_net, dec_net, loss, cost, opt


inputs_ = tf.placeholder(tf.float32, (None, 28, 28, 1), name='inputs')
targets_ = tf.placeholder(tf.float32, (None, 28, 28, 1), name='targets')

batch_size = 200  # Number of samples in each batch
epoch_num = 100     # Number of epochs to train the network
lr = 0.001        # Learning rate

encoded_net, decoded_net, loss, cost, opt = conv_autoencoder(inputs_, targets_, lr)

# read MNIST dataset
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

# calculate the number of batches per epoch
batch_per_ep = mnist.train.num_examples // batch_size

saver = tf.train.Saver()
sess = tf.Session()

# Set's how much noise we're adding to the MNIST images
noise_factor = 0.5
sess.run(tf.global_variables_initializer())
for e in range(epoch_num):
    for ii in range(batch_per_ep):
        batch_img, batch_label = mnist.train.next_batch(batch_size)  # read a batch        # Get images from the batch
        imgs = batch_img[0].reshape((-1, 28, 28, 1))
        
        # Add random noise to the input images
        noisy_imgs = imgs + noise_factor * np.random.randn(*imgs.shape)
        # Clip the images to be between 0 and 1
        noisy_imgs = np.clip(noisy_imgs, 0., 1.)
        
        # Noisy images as inputs, original images as targets
        batch_cost, _ = sess.run([cost, opt], feed_dict={inputs_: noisy_imgs,
                                                         targets_: imgs})

    print("Epoch: {}/{}...".format(e+1, epoch_num), "Training loss: {:.4f}".format(batch_cost))

save_path = saver.save(sess,"/tmp/model_simple.ckpt")
print("Model saved in path: %s" % save_path)

"""
saver.restore(sess,"/tmp/model_simple.ckpt")
print("Model restored")
"""
# test the trained network
img, batch_label = mnist.test.next_batch(50)
recon_img = sess.run([decoded_net], feed_dict={inputs_: img})[0]
encoded_img = sess.run([encoded_net], feed_dict={inputs_: img})[0]

# plot the reconstructed images and their ground truths (inputs)
plt.figure(1)
plt.title('Reconstructed Images')
for i in range(50):
    plt.subplot(5, 10, i+1)
    plt.imshow(recon_img[i, ..., 0], cmap='gray')
plt.figure(2)
plt.title('Input Images')
for i in range(50):
    plt.subplot(5, 10, i+1)
    plt.imshow(batch_img[i, ..., 0], cmap='gray')
plt.figure(3)
plt.title('Encoded Images')
for i in range(50):
    plt.subplot(5, 10, i+1)
    plt.imshow(encoded_img[i, ..., 0], cmap='gray')
plt.show()