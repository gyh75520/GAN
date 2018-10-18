import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


def sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])


X = tf.placeholder(tf.float32, [None, 28, 28, 1])
Z = tf.placeholder(tf.float32, [None, 62])

w_init = tf.random_normal_initializer(mean=0, stddev=0.02)
b_init = tf.constant_initializer(0)


def generator():
    with tf.variable_scope("generator"):
        # z = [batch,62]
        # G_dense1 = slim.fully_connected(Z, 1024, activation_fn=tf.nn.relu, weights_initializer=w_init, biases_initializer=b_init)
        # G_dense1 = [batch,1024]
        G_dense2 = slim.fully_connected(Z, 128 * 7 * 7, activation_fn=tf.nn.relu, weights_initializer=w_init, biases_initializer=b_init)
        # G_dense1 = [batch,128 * 7 * 7]
        G_dense2_flatt = tf.reshape(G_dense2, [-1, 7, 7, 128])
        G_deconv1 = slim.conv2d_transpose(G_dense2_flatt, 64, [4, 4], 2, activation_fn=tf.nn.relu, weights_initializer=w_init, biases_initializer=b_init)
        # G_deconv1 = [batch,14,14,64]
        G_deconv2 = slim.conv2d_transpose(G_deconv1, 1, [4, 4], 2, activation_fn=None, weights_initializer=w_init, biases_initializer=b_init)
        # G_deconv2 = [batch,28,28,1]
        G_logits = G_deconv2
        G_prob = tf.nn.sigmoid(G_logits)
        return G_prob


def discriminator(input, reuse=False):
    with tf.variable_scope("discriminator", reuse=reuse):
        # input = [batch,28,28,1]
        D_conv1 = slim.conv2d(input, 64, [4, 4], 2, padding="SAME", activation_fn=tf.nn.relu, weights_initializer=w_init, biases_initializer=b_init, reuse=reuse, scope="d_conv1")
        # D_conv1 = [batch,14,14,64]
        D_conv2 = slim.conv2d(D_conv1, 128, [4, 4], 2, padding="SAME", activation_fn=tf.nn.relu,
                              weights_initializer=w_init, biases_initializer=b_init, reuse=reuse, scope="d_conv2")
        # D_conv1 = [batch,7,7,128]
        flat_size = D_conv2.shape[1] * D_conv2.shape[2] * D_conv2.shape[3]
        D_conv2_flat = tf.reshape(D_conv2, [-1, int(flat_size)])
        # D_dense1 = slim.fully_connected(D_conv2_flat, 1024, activation_fn=tf.nn.relu, weights_initializer=w_init, biases_initializer=b_init, reuse=reuse, scope="d_dense1")
        D_dense2 = slim.fully_connected(D_conv2_flat, 1, activation_fn=None, weights_initializer=w_init, biases_initializer=b_init, reuse=reuse, scope="d_dense2")
        # D_dense2 = [batch,1]
        D_logits = D_dense2
        D_prob = tf.nn.sigmoid(D_logits)
        return D_logits, D_prob


def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig


G_sample = generator()
D_logits_real, _ = discriminator(X)
D_logits_fake, _ = discriminator(G_sample, reuse=True)

D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logits_real, labels=tf.ones_like(D_logits_real)))
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logits_fake, labels=tf.zeros_like(D_logits_fake)))
D_loss = D_loss_real + D_loss_fake

G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logits_fake, labels=tf.ones_like(D_logits_fake)))

t_var = tf.trainable_variables()
theta_D = [var for var in t_var if 'discriminator' in var.name]
theta_G = [var for var in t_var if 'generator' in var.name]

print("theta_D   ", theta_D)
print("theta_G   ", theta_G)
# print(theta_G)
D_optimizer = tf.train.AdamOptimizer(learning_rate=2e-4).minimize(D_loss, var_list=theta_D)  # note:only update D gradient, use var_list
G_optimizer = tf.train.AdamOptimizer(learning_rate=2e-4).minimize(G_loss, var_list=theta_G)  # note:only update G gradient, use var_list

mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)


os.environ['CUDA_VISIBLE_DEVICES'] = '1'
gpuConfig = tf.ConfigProto(allow_soft_placement=True)
gpuConfig.gpu_options.allow_growth = True

with tf.Session(config=gpuConfig) as sess:
    sess.run(tf.global_variables_initializer())
    dir_name = os.path.basename(__file__).split('.')[0] + '_result'
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    i = 0
    mb_size = 128
    Z_dim = 62

    z_show = sample_Z(16, Z_dim)
    for it in range(1000000):
        if it % 100 == 0:
            samples = sess.run(G_sample, feed_dict={Z: z_show})

            fig = plot(samples)
            plt.savefig(dir_name + '/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
            i += 1
            plt.close(fig)

        X_mb, _ = mnist.train.next_batch(mb_size)
        X_mb = np.reshape(X_mb, [-1, 28, 28, 1])
        _, D_loss_curr = sess.run([D_optimizer, D_loss], feed_dict={X: X_mb, Z: sample_Z(mb_size, Z_dim)})
        _, G_loss_curr = sess.run([G_optimizer, G_loss], feed_dict={Z: sample_Z(mb_size, Z_dim)})

        if it % 100 == 0:
            print('Iter: {}'.format(it))
            print('D loss: {:.4}'. format(D_loss_curr))
            print('G_loss: {:.4}'.format(G_loss_curr))
            print()
