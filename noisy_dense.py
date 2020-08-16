import numpy as np
import tensorflow as tf


def sample_noise(shape):
    noise = tf.random_normal(shape)
    return noise

def noisy_dense(x, size, name, evaluation, bias=True, activation_fn=tf.identity, sigma_zero=0.5):
    mu_init = tf.random_uniform_initializer(minval=-1 * 1 / np.power(x.get_shape().as_list()[1], 0.5),
                                            maxval=1 * 1 / np.power(x.get_shape().as_list()[1], 0.5))

    sigma_init = tf.constant_initializer(sigma_zero / np.power(x.get_shape().as_list()[1], 0.5))

    p = sample_noise([x.get_shape().as_list()[1], 1])
    q = sample_noise([1, size])

    f_p = tf.multiply(tf.sign(p), tf.pow(tf.abs(p), 0.5))
    f_q = tf.multiply(tf.sign(q), tf.pow(tf.abs(q), 0.5))

    w_epsilon = f_p * f_q
    b_epsilon = tf.squeeze(f_q)

    w_mu = tf.get_variable(name + "/w_mu", [x.get_shape()[1], size], initializer=mu_init)
    w_sigma = tf.get_variable(name + "/w_sigma", [x.get_shape()[1], size], initializer=sigma_init)
    w = tf.cond(tf.equal(evaluation, tf.constant(True)), lambda: w_mu, lambda: w_mu + tf.multiply(w_sigma, w_epsilon))

    ret = tf.matmul(x, w)
    if bias:
        b_mu = tf.get_variable(name + "/bias_mu", [size], initializer=mu_init)
        b_sigma = tf.get_variable(name + "/bias_sigma", [size], initializer=sigma_init)
        b = tf.cond(tf.equal(evaluation, tf.constant(True)), lambda: b_mu,
                    lambda: b_mu + tf.multiply(b_sigma, b_epsilon))

        return activation_fn(ret + b), w_sigma, b_sigma
    else:
        return activation_fn(ret)