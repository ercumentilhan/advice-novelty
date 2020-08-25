import numpy as np
import tensorflow as tf
from constants import *
from noisy_dense import noisy_dense


class RND(object):
    def __init__(self, id, config, session, lr):
        self.id = id + '_RND'
        self.session = session

        # Extract relevant configuration:
        self.config = {}
        self.config['env_name'] = config['env_name']
        self.config['env_n_actions'] = config['env_n_actions']
        self.config['env_obs_dims'] = config['env_obs_dims']
        self.config['env_obs_form'] = config['env_obs_form']
        self.config['experiment_setup'] = config['experiment_setup']
        self.config['n_training_frames'] = config['n_training_frames']

        # Hyperparameters
        self.config['rnd_learning_rate'] = lr

        # Hardcoded hyperparameters
        self.config['rnd_adam_epsilon'] = 0.00015
        self.config['rnd_output_size'] = 6
        self.config['rnd_hidden_size'] = 128

        normalization_coefficients = self.get_normalization_coefficients(self.config['env_name'])
        self.obs_mean = np.clip(np.array(normalization_coefficients[0]), a_min=0.00001, a_max=None)
        self.obs_std = np.clip(np.array(normalization_coefficients[1]), a_min=0.00001, a_max=None)

        self.name_fixed = self.id + '/RND_FIXED'
        self.name_online = self.id + '/RND_ONLINE'

        self.input_fixed, self.output_fixed, self.evaluation_fixed = \
            self.build_model(self.name_fixed, self.config['rnd_hidden_size'], self.config['rnd_output_size'])

        self.input_online, self.output_online, self.evaluation_online = \
            self.build_model(self.name_online, self.config['rnd_hidden_size'], self.config['rnd_output_size'])

        self.labels, \
        self.losses, \
        self.minimises, \
        self.error = self.build_training_op()

        self.training_steps = 0

    # ------------------------------------------------------------------------------------------------------------------

    def dense_net(self, scope, inputs, hidden_size, output_size):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            layer_1 = tf.layers.dense(inputs, hidden_size, use_bias=True,
                                      kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                      activation=tf.nn.relu, name='DENSE_LAYER_1')

            layer_2 = tf.layers.dense(layer_1, output_size, use_bias=True,
                                      kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                      activation=None, name='DENSE_LAYER_2')

            return layer_2

    def noisy_dense_net(self, scope, inputs, hidden_size, output_size, evaluation):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            layer_1, _, _ = noisy_dense(inputs, size=hidden_size, bias=True, evaluation=evaluation,
                                        activation_fn=tf.nn.relu, name='N_DENSE_LAYER_1')

            layer_2, _, _ = noisy_dense(layer_1, size=output_size, bias=True, evaluation=evaluation,
                                        name='N_DENSE_LAYER_2')
            return layer_2

    def convolutional_net(self, scope, inputs):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            layer_1 = tf.layers.conv2d(inputs=inputs,
                                       filters=16,
                                       kernel_size=(3, 3),
                                       strides=(1, 1),
                                       padding='VALID',
                                       kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                       activation=tf.nn.relu,
                                       name='CONV_LAYER_1')

            layer_out = tf.contrib.layers.flatten(layer_1)
            print(layer_out.get_shape())
            return layer_out

    # ------------------------------------------------------------------------------------------------------------------

    def build_model(self, name, dense_hidden_size, output_size):
        input = None
        if self.config['env_obs_form'] == NONSPATIAL:
            input = tf.placeholder(tf.float32, [None, self.config['env_obs_dims'][0]], name=name + '_OBS')
        elif self.config['env_obs_form'] == SPATIAL:
            input = tf.placeholder(tf.float32, [None, self.config['env_obs_dims'][0],
                                              self.config['env_obs_dims'][1],
                                              self.config['env_obs_dims'][2]], name=name + '_OBS')

        evaluation = tf.placeholder(tf.bool, name=name + '_EVALUATION')

        latent_features = None
        if self.config['env_obs_form'] == NONSPATIAL:
            latent_features = input
        elif self.config['env_obs_form'] == SPATIAL:
            latent_features = self.convolutional_net(name, input)

        output = self.dense_net(name, latent_features, dense_hidden_size, output_size)

        return input, output, evaluation

    # ------------------------------------------------------------------------------------------------------------------

    def build_training_op(self):
        label = tf.placeholder(tf.float32, [None, self.config['rnd_output_size']], name='LABELS_' + str(self.id))
        error = tf.abs(label - self.output_online)

        optimizer = tf.train.AdamOptimizer(self.config['rnd_learning_rate'], epsilon=self.config['rnd_adam_epsilon'])

        losses = []
        losses.append(tf.losses.mean_squared_error(labels=label, predictions=self.output_online))

        minimises = []
        for loss in losses:
            minimises.append(optimizer.minimize(loss))

        return label, losses, minimises, error

    # ------------------------------------------------------------------------------------------------------------------

    def train_model(self, obs_batch_in, loss_id, is_batch=True, normalize=True):
        self.training_steps += 1

        if normalize:
            obs_batch_in = self.normalize_obs(obs_batch_in)

        obs_batch = obs_batch_in if isinstance(obs_batch_in, list) else obs_batch_in
        obs_batch = obs_batch if is_batch else [obs_batch]

        feed_dict = {self.input_fixed: obs_batch, self.evaluation_fixed: False}

        output_fixed = self.session.run([self.output_fixed], feed_dict=feed_dict)[0]

        feed_dict = {self.input_online: obs_batch, self.evaluation_online: False,
                     self.labels: output_fixed}

        loss, _, error = self.session.run([self.losses[loss_id], self.minimises[loss_id], self.error],
                                          feed_dict=feed_dict)

        return loss, np.abs(error)

    # ------------------------------------------------------------------------------------------------------------------

    def get_error(self, obs_in, evaluation=False, normalize=True):

        obs = self.normalize_obs(obs_in) if normalize else obs_in

        feed_dict = {self.input_fixed: [obs],
                     self.input_online: [obs],
                     self.evaluation_fixed: evaluation,
                     self.evaluation_online: evaluation}

        output_fixed, output_online = self.session.run([self.output_fixed, self.output_online], feed_dict=feed_dict)

        return ((output_fixed - output_online) ** 2).mean(axis=1)[0]

    # ------------------------------------------------------------------------------------------------------------------

    def normalize_obs(self, obs):
        return np.clip(((obs - self.obs_mean) / self.obs_std), -5, 5)

    # ------------------------------------------------------------------------------------------------------------------

    def get_normalization_coefficients(self, env_name):
        return RND_MEAN_COEFFS[env_name], RND_STD_COEFFS[env_name]
