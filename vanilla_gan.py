import tensorflow as tf
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


X = tf.placeholder(tf.float32, [None, 784])
D_w1 = tf.Variable(xavier_init([784, 128]))
D_b1 = tf.Variable(tf.zeros([1, 128]))
D_w2 = tf.Variable(xavier_init([128, 1]))
D_b2 = tf.Variable(tf.zeros([1, 1]))

theta_D = [D_w1, D_w2, D_b1, D_b2]

Z = tf.placeholder(tf.float32, [None, 100])
G_w1 = tf.Variable(xavier_init([100, 128]))
G_b1 = tf.Variable(tf.zeros([1, 128]))
G_w2 = tf.Variable(xavier_init([128, 784]))
G_b2 = tf.Variable(tf.zeros([1, 784]))

theta_G = [G_w1, G_w2, G_b1, G_b2]


def generator():
    G_h1 = tf.nn.relu(tf.matmul(Z, G_w1) + G_b1)
    G_logits = tf.matmul(G_h1, G_w2) + G_b2
    G_prob = tf.nn.sigmoid(G_logits)
    return G_prob


def discriminator(input):
    D_h1 = tf.nn.relu(tf.matmul(input, D_w1) + D_b1)
    D_logits = tf.matmul(D_h1, D_w2) + D_b2
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
D_logits_fake, _ = discriminator(G_sample)

D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logits_real, labels=tf.ones_like(D_logits_real)))
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logits_fake, labels=tf.zeros_like(D_logits_fake)))
D_loss = D_loss_real + D_loss_fake

G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logits_fake, labels=tf.ones_like(D_logits_fake)))

D_optimizer = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)  # note:only update D gradient, use var_list
G_optimizer = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)  # note:only update G gradient, use var_list


mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)


os.environ['CUDA_VISIBLE_DEVICES'] = '1'
gpuConfig = tf.ConfigProto(allow_soft_placement=True)
gpuConfig.gpu_options.allow_growth = True

with tf.Session(config=gpuConfig) as sess:
    # dir_name = "vanilla_gan_ckpt"
    # sess.run(tf.global_variables_initializer())
    # saver = tf.train.Saver()
    # ckpt = tf.train.get_checkpoint_state(dir_name)
    # if ckpt and ckpt.model_checkpoint_path:
    #     saver.restore(sess, ckpt.model_checkpoint_path)
    #     print("\nrestore success!")
    # batch_size = 128
    #
    # for i in range(0, 1000000):
    #     X_mb, _ = mnist.train.next_batch(batch_size)
    #     # plt.imshow(X_mb[0].reshape(28, 28))
    #     # plt.show()
    #     # from scipy import misc
    #     # misc.imsave("2.png", X_mb[0].reshape(28, 28))
    #     _, D_loss_curr = sess.run([D_optimizer, D_loss], feed_dict={X: X_mb, Z: sample_Z(batch_size, 100)})
    #     _, G_loss_curr = sess.run([G_optimizer, G_loss], feed_dict={Z: sample_Z(batch_size, 100)})
    #     if i % 1000 == 0:
    #         samples = sess.run(G_sample, feed_dict={Z: sample_Z(1, 100)})
    #         from scipy import misc
    #         misc.imsave(str(i) + ".png", samples[0].reshape(28, 28))
    #         # plt.savefig(str(i) + ".png")
    #         # print(type(samples[0]))
    #         print("D_loss: ", D_loss_curr)
    #         print("G_loss: ", G_loss_curr)
    #
    #         saver.save(sess, dir_name + '/mnist.ckpt', global_step=i)
    #
    # print("finish")

    sess.run(tf.global_variables_initializer())

    dir_name = os.path.basename(__file__).split('.')[0] + '_result'
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    i = 0
    mb_size = 128
    Z_dim = 100
    for it in range(1000000):
        if it % 1000 == 0:
            samples = sess.run(G_sample, feed_dict={Z: sample_Z(16, Z_dim)})

            fig = plot(samples)
            plt.savefig(dir_name + '/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
            i += 1
            plt.close(fig)

        X_mb, _ = mnist.train.next_batch(mb_size)

        _, D_loss_curr = sess.run([D_optimizer, D_loss], feed_dict={X: X_mb, Z: sample_Z(mb_size, Z_dim)})
        _, G_loss_curr = sess.run([G_optimizer, G_loss], feed_dict={Z: sample_Z(mb_size, Z_dim)})

        if it % 1000 == 0:
            print('Iter: {}'.format(it))
            print('D loss: {:.4}'. format(D_loss_curr))
            print('G_loss: {:.4}'.format(G_loss_curr))
            print()
