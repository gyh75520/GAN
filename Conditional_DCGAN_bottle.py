'''
Architecture guidelines for stable Deep Convolutional GANs :

Replace any pooling layers with strided convolutions(discriminator) and fractional - strided convolutions(generator).
Use batchnorm in both the generator and the discriminator
Remove fully connected hidden layers for deeper architectures. Just use average pooling at the end.
Use ReLU activation in generator for all layers except for the output, which uses Tanh.
Use LeakyReLU activation in the discriminator for all layers.

Conditional GAN :
Apply GAN by adding condition(supervised)
y: the condition to constranid noise to generate a image
G(z)--> G(z,y) , Z(X)-->Z(X,y)

'''
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datas import get_batch


def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


def sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])


def show_Y(m):
    Y_mb, _ = get_batch(m, crop_img=False, magnify_interval=False)
    return Y_mb


def G_concat(z, y):
    batch_size = tf.shape(z)[0]
    z = tf.reshape(z, [batch_size, 1, 1, 100])
    return tf.concat([y, z * tf.ones([batch_size, 480, 640, 100])], 3)


def D_concat(x, y):
    batch_size = tf.shape(x)[0]
    x = tf.reshape(x, [batch_size, 1, 1, 32])
    # print(tf.concat([y, x * tf.ones([batch_size, 480, 640, 4])], 3))
    return tf.concat([y, x * tf.ones([batch_size, 480, 640, 32])], 3)


X = tf.placeholder(tf.float32, [None, 32])
Y = tf.placeholder(tf.float32, [None, 480, 640, 3])
Z = tf.placeholder(tf.float32, [None, 100])


w_init = tf.random_normal_initializer(mean=0, stddev=0.02)
b_init = tf.constant_initializer(0)


def generator(z):
    with tf.variable_scope("generator"):
        # z = [batch,62]
        # input = [batch,64,64,3]
        G_conv1 = slim.conv2d(z, 64, [4, 4], 2, padding="SAME", activation_fn=tf.nn.leaky_relu, weights_initializer=w_init, biases_initializer=b_init)
        # D_conv1 = [batch,32,32,64]
        G_conv2 = slim.conv2d(G_conv1, 64 * 2, [4, 4], 2, padding="SAME", activation_fn=tf.nn.leaky_relu, weights_initializer=w_init, normalizer_fn=slim.batch_norm)
        # D_conv2 = [batch,16,16,128]
        G_conv3 = slim.conv2d(G_conv2, 64 * 4, [4, 4], 2, padding="SAME", activation_fn=tf.nn.leaky_relu, weights_initializer=w_init, normalizer_fn=slim.batch_norm)
        # D_conv2 = [batch,8,8,256]
        G_conv4 = slim.conv2d(G_conv3, 64 * 8, [4, 4], 2, padding="SAME", activation_fn=tf.nn.leaky_relu, weights_initializer=w_init, normalizer_fn=slim.batch_norm)
        # D_conv2 = [batch,4,4,512]
        print('G_conv4', G_conv4)
        flat_size = G_conv4.shape[1] * G_conv4.shape[2] * G_conv4.shape[3]
        G_conv4_flat = tf.reshape(G_conv4, [-1, int(flat_size)])
        # D_dense1 = slim.fully_connected(D_conv2_flat, 1024, activation_fn=tf.nn.relu, weights_initializer=w_init, normalizer_fn=slim.batch_norm)
        G_dense2 = slim.fully_connected(G_conv4_flat, 32, activation_fn=None, weights_initializer=w_init, biases_initializer=b_init)
        # D_dense2 = [batch,1]
        G_logits = G_dense2
        G_prob = tf.nn.sigmoid(G_logits)
        return G_prob


def discriminator(input, reuse=False):
    with tf.variable_scope("discriminator", reuse=reuse):
        # input = [batch,64,64,3]
        D_conv1 = slim.conv2d(input, 64, [4, 4], 2, padding="SAME", activation_fn=tf.nn.leaky_relu, weights_initializer=w_init, biases_initializer=b_init)
        # D_conv1 = [batch,32,32,64]
        D_conv2 = slim.conv2d(D_conv1, 64 * 2, [4, 4], 2, padding="SAME", activation_fn=tf.nn.leaky_relu, weights_initializer=w_init, normalizer_fn=slim.batch_norm)
        # D_conv2 = [batch,16,16,128]
        D_conv3 = slim.conv2d(D_conv2, 64 * 4, [4, 4], 2, padding="SAME", activation_fn=tf.nn.leaky_relu, weights_initializer=w_init, normalizer_fn=slim.batch_norm)
        # D_conv2 = [batch,8,8,256]
        D_conv4 = slim.conv2d(D_conv3, 64 * 8, [4, 4], 2, padding="SAME", activation_fn=tf.nn.leaky_relu, weights_initializer=w_init, normalizer_fn=slim.batch_norm)
        # D_conv2 = [batch,4,4,512]
        print('D_conv3', D_conv3)
        flat_size = D_conv4.shape[1] * D_conv4.shape[2] * D_conv4.shape[3]
        D_conv4_flat = tf.reshape(D_conv4, [-1, int(flat_size)])
        # D_dense1 = slim.fully_connected(D_conv2_flat, 1024, activation_fn=tf.nn.relu, weights_initializer=w_init, normalizer_fn=slim.batch_norm)
        D_dense2 = slim.fully_connected(D_conv4_flat, 1, activation_fn=None, weights_initializer=w_init, biases_initializer=b_init)
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
        plt.imshow(sample)

    return fig


G_sample = generator(G_concat(Z, Y))
D_logits_real, _ = discriminator(D_concat(X, Y))
D_logits_fake, _ = discriminator(D_concat(G_sample, Y), reuse=True)

D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logits_real, labels=tf.ones_like(D_logits_real)))
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logits_fake, labels=tf.zeros_like(D_logits_fake)))
D_loss = D_loss_real + D_loss_fake

G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logits_fake, labels=tf.ones_like(D_logits_fake)))

t_var = tf.trainable_variables()
theta_D = [var for var in t_var if 'discriminator' in var.name]
theta_G = [var for var in t_var if 'generator' in var.name]

theta_D_new = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
print("theta_D   ", theta_D)
print("theta_D_new   ", theta_D_new)
# print(theta_G)


D_optimizer = tf.train.AdamOptimizer(learning_rate=2e-4).minimize(D_loss, var_list=theta_D)  # note:only update D gradient, use var_list
G_optimizer = tf.train.AdamOptimizer(learning_rate=2e-4).minimize(G_loss, var_list=theta_G)  # note:only update G gradient, use var_list


os.environ['CUDA_VISIBLE_DEVICES'] = '1'
gpuConfig = tf.ConfigProto(allow_soft_placement=True)
gpuConfig.gpu_options.allow_growth = True

with tf.Session(config=gpuConfig) as sess:
    sess.run(tf.global_variables_initializer())
    dir_name = os.path.basename(__file__).split('.')[0] + '_result'
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    i = 0
    mb_size = 32
    Z_dim = 100

    z_show = sample_Z(16, Z_dim)
    y_show = show_Y(16)

    for it in range(1000000):
        if it % 100 == 0:
            samples = sess.run(G_sample, feed_dict={Y: y_show, Z: z_show})
            samples = (samples + 1.) / 2.  # inverse transform from [-1,1] to [0,1]
            fig = plot(samples)
            plt.savefig(dir_name + '/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
            i += 1
            plt.close(fig)

        Y_mb, X_mb = get_batch(mb_size, crop_img=False, magnify_interval=False)
        # X_mb = np.reshape(X_mb, [-1, 28, 28, 1])
        _, D_loss_curr = sess.run([D_optimizer, D_loss], feed_dict={X: X_mb, Y: Y_mb, Z: sample_Z(mb_size, Z_dim)})
        _, G_loss_curr = sess.run([G_optimizer, G_loss], feed_dict={Y: Y_mb, Z: sample_Z(mb_size, Z_dim)})

        if it % 100 == 0:
            print('Iter: {}'.format(it))
            print('D loss: {:.4}'. format(D_loss_curr))
            print('G_loss: {:.4}'.format(G_loss_curr))
            print()
