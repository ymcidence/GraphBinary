import tensorflow as tf
from tensorflow.python.framework import function
from util.layer import conventional_layers as layers
from util.layer import graph_conv as gcn

D_TYPE = tf.float32


@function.Defun(D_TYPE, D_TYPE, D_TYPE, D_TYPE)
def doubly_sn_grad(logits, epsilon, dprev, dpout):
    prob = 1.0 / (1 + tf.exp(-logits))
    # yout = (tf.sign(prob - epsilon) + 1.0) / 2.0

    dlogits = prob * (1 - prob) * (dprev + dpout)

    depsilon = dprev
    return dlogits, depsilon


@function.Defun(D_TYPE, D_TYPE, grad_func=doubly_sn_grad)
def doubly_sn(logits, epsilon):
    prob = 1.0 / (1 + tf.exp(-logits))
    yout = (tf.sign(prob - epsilon) + 1.0) / 2.0
    return yout, prob


class BasicModel(object):
    def __init__(self, **kwargs):
        self.batch_size = kwargs.get('batch_size', 100)
        self.code_length = kwargs.get('code_length', 32)
        self.image_in = tf.placeholder(dtype=D_TYPE, shape=[None, None])
        self.lam = kwargs.get('lam', 1)
        self.net = self._build_net()

    def _build_net(self):
        with tf.variable_scope('encoder'):
            fc_1 = layers.fc_relu_layer('fc_1', bottom=self.image_in, output_dim=512)
            # continuous_hidden = tf.nn.sigmoid(layers.fc_layer('cont_hidden', bottom=fc_1, output_dim=256))

            _batch_size, _feature_size = self.image_in.get_shape().as_list()

            eps = tf.ones([_batch_size, self.code_length], dtype=D_TYPE) * 0.5
            code_hidden = tf.nn.sigmoid(layers.fc_layer('code_hidden', bottom=fc_1, output_dim=self.code_length))

            codes, code_prob = doubly_sn(code_hidden, eps)

            batch_adjacency = gcn.build_adjacency(codes)

            continuous_hidden = gcn.spectrum_conv_layer('gcn', fc_1, batch_adjacency, 256)

        with tf.variable_scope('decoder'):
            fc_2 = layers.fc_relu_layer('fc_2', continuous_hidden, 512)
            decode_result = layers.fc_layer('decode_result', fc_2, _feature_size)

        with tf.variable_scope('critic'):
            real_logic = tf.sigmoid(layers.fc_layer('critic', bottom=continuous_hidden, output_dim=1),
                                    name='critic_sigmoid')

        with tf.variable_scope('critic', reuse=True):
            random_in = tf.random.uniform([_batch_size, 256])
            fake_logic = tf.sigmoid(layers.fc_layer('critic', bottom=random_in, output_dim=1),
                                    name='critic_sigmoid')

        return {
            'codes': codes,
            'code_hidden': code_hidden,
            'decode_result': decode_result,
            'hidden': continuous_hidden,
            'real_logic': real_logic,
            'fake_logic': fake_logic}

    def _build_loss(self):
        critic_loss = tf.reduce_sum(tf.log(self.net.get('fake_logic'))
                                    + tf.log(1 - self.net.get('real_logic'))) * -1 * self.lam

        q_zx = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.net.get('code_hidden'),
                                                                      labels=self.net.get('codes')))
        encoding_loss = tf.nn.l2_loss(self.net.get('decode_result') - self.image_in) - self.lam * tf.reduce_sum(
            tf.log(self.net.get('real_logic'))) + q_zx

        return encoding_loss, critic_loss
