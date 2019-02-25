import tensorflow as tf
from tensorflow.python.framework import function
from util.layer import conventional_layers as layers
from util.layer import graph_conv as gcn
from util.data.dataset import DataHelper, MatDataset
from time import gmtime, strftime
import os

D_TYPE = tf.float32


def loss_regu(par_list, weight=0.):
    single_regu = [tf.nn.l2_loss(v) for v in par_list]
    loss = tf.add_n(single_regu) * weight
    return loss


def fc_layer_hack(name, bottom, input_dim, output_dim, bias_term=True, weights_initializer=None,
                  biases_initializer=None):
    # flatten bottom input
    # input has shape [batch, in_height, in_width, in_channels]
    flat_bottom = bottom

    # weights and biases variables
    with tf.variable_scope(name):
        # initialize the variables
        if weights_initializer is None:
            weights_initializer = tf.random_normal_initializer(stddev=0.01)
        if bias_term and biases_initializer is None:
            biases_initializer = tf.constant_initializer(0.)

        # weights has shape [input_dim, output_dim]
        weights = tf.get_variable("kernel", [input_dim, output_dim], initializer=weights_initializer)
        if bias_term:
            biases = tf.get_variable("bias", output_dim, initializer=biases_initializer)
            fc = tf.nn.xw_plus_b(flat_bottom, weights, biases)
        else:
            fc = tf.matmul(flat_bottom, weights)
    return fc


@function.Defun(D_TYPE, D_TYPE, D_TYPE, D_TYPE)
def doubly_sn_grad(logits, epsilon, dprev, dpout):
    prob = 1.0 / (1 + tf.exp(-logits))
    yout = (tf.sign(prob - epsilon) + 1.0) / 2.0

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
        self.image_in = tf.placeholder(dtype=D_TYPE, shape=[self.batch_size, 4096])
        self.lam = kwargs.get('lam', 1)
        self.net = self._build_net()
        self.global_step = tf.Variable(0, trainable=False, name='global_step')

    def _build_net(self):
        with tf.variable_scope('actor'):
            with tf.variable_scope('encoder'):
                fc_1 = layers.fc_relu_layer('fc_1', bottom=self.image_in, output_dim=2048)
                # continuous_hidden = tf.nn.sigmoid(layers.fc_layer('cont_hidden', bottom=fc_1, output_dim=256))

                _batch_size, _feature_size = self.image_in.get_shape().as_list()

                eps = tf.ones([_batch_size, self.code_length], dtype=D_TYPE) * 0.5
                code_hidden = layers.fc_layer('code_hidden', bottom=fc_1, output_dim=self.code_length)

                codes, code_prob = doubly_sn(code_hidden, eps)

                batch_adjacency = gcn.build_adjacency_hamming(codes, code_length=self.code_length)

                continuous_hidden = tf.nn.sigmoid(
                    gcn.spectrum_conv_layer('gcn', fc_1, batch_adjacency, 512, _batch_size))

            with tf.variable_scope('decoder'):
                fc_2 = layers.fc_relu_layer('fc_2', continuous_hidden, 2048)
                decode_result = layers.fc_layer('decode_result', fc_2, _feature_size)

        with tf.variable_scope('critic'):
            real_logic = tf.sigmoid(layers.fc_layer('critic', bottom=continuous_hidden, output_dim=1),
                                    name='critic_sigmoid')

            real_binary_logic = tf.sigmoid(
                fc_layer_hack('critic_2', bottom=codes, input_dim=self.code_length, output_dim=1),
                name='critic_sigmoid_2')

        with tf.variable_scope('critic', reuse=True):
            random_in = tf.random.uniform([_batch_size, 512])
            fake_logic = tf.sigmoid(layers.fc_layer('critic', bottom=random_in, output_dim=1),
                                    name='critic_sigmoid')

            random_binary = (tf.sign(tf.random.uniform([_batch_size, self.code_length]) - 0.5) + 1) / 2
            fake_binary_logic = tf.sigmoid(
                fc_layer_hack('critic_2', bottom=random_binary, input_dim=self.code_length, output_dim=1),
                name='critic_sigmoid_2')

        adj_pic = tf.expand_dims(tf.expand_dims(batch_adjacency, axis=0), axis=-1)

        tf.summary.image('actor/adj', adj_pic)

        tf.summary.histogram('critic/fake_cont', random_in)
        tf.summary.histogram('critic/fake_binary', random_binary)
        tf.summary.scalar('critic/fake_logic', tf.reduce_mean(fake_logic))
        tf.summary.scalar('actor/real_logic', tf.reduce_mean(real_logic))
        tf.summary.histogram('actor/real_cont', continuous_hidden)
        tf.summary.histogram('actor/binary', codes)
        tf.summary.histogram('actor/code_hidden', code_hidden)
        tf.summary.scalar('actor/binary', tf.reduce_mean(codes))

        return {
            'codes': codes,
            'code_hidden': code_hidden,
            'decode_result': decode_result,
            'hidden': continuous_hidden,
            'real_logic': real_logic,
            'fake_logic': fake_logic,
            'real_binary_logic': real_binary_logic,
            'fake_binary_logic': fake_binary_logic}

    def _build_loss(self):
        critic_regu = loss_regu(tf.trainable_variables(scope='critic'))
        critic_loss_1 = tf.reduce_mean(tf.log(self.net.get('fake_logic'))
                                       + tf.log(1 - self.net.get('real_logic'))) * -1 * self.lam

        critic_loss_2 = tf.reduce_mean(
            tf.log(self.net.get('fake_binary_logic')) + tf.log(1 - self.net.get('real_binary_logic'))) * -1 * self.lam

        critic_loss = critic_loss_1 + critic_loss_2 + critic_regu

        # q_zx = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.net.get('code_hidden'),
        #                                                               labels=self.net.get('codes')))

        actor_regu = loss_regu(tf.trainable_variables(scope='actor'))
        encoding_loss = tf.reduce_mean(tf.nn.l2_loss(
            self.net.get('decode_result') - self.image_in)) - self.lam * tf.reduce_mean(
            tf.log(self.net.get('real_logic'))) - self.lam * tf.reduce_mean(
            tf.log(self.net.get('real_binary_logic'))) + actor_regu

        tf.summary.scalar('critic/loss', critic_loss)
        tf.summary.scalar('critic/regu', critic_regu)
        tf.summary.scalar('actor/loss', encoding_loss)
        tf.summary.scalar('actor/regu', actor_regu)
        # tf.summary.scalar('actor/qzx', q_zx)

        return encoding_loss, critic_loss

    def opt(self, operator: tf.Tensor, scope=None):
        train_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
        return tf.train.AdamOptimizer(1e-4).minimize(operator, global_step=self.global_step, var_list=train_list)

    def train(self, sess: tf.Session, data: DataHelper, restore_file=None):
        actor_loss, critic_loss = self._build_loss()
        actor_opt = self.opt(actor_loss, 'actor')
        critic_opt = self.opt(critic_loss, 'critic')
        initial_op = tf.global_variables_initializer()

        sess.run(initial_op)

        time_string = strftime("%a%d%b%Y-%H%M%S", gmtime())
        summary_path = os.path.join('data', 'log', time_string) + os.sep
        save_path = os.path.join('data', 'model') + os.sep

        if not os.path.exists(save_path):
            os.mkdir(save_path)

        if restore_file is not None:
            self._restore(sess, restore_file)

        writer = tf.summary.FileWriter(summary_path)

        actor_summary = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES, scope='actor'))
        critic_summary = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES, scope='critic'))

        for i in range(50000):
            train_batch = data.next_batch('train')
            train_dict = {self.image_in: train_batch['batch_image']}
            _, critic_value, critic_summary_value, critic_step = sess.run(
                [critic_opt, critic_loss, critic_summary, self.global_step], feed_dict=train_dict)
            _, actor_value, actor_summary_value, code_value, actor_step = sess.run(
                [actor_opt, actor_loss, actor_summary, self.net['codes'], self.global_step], feed_dict=train_dict)

            writer.add_summary(critic_summary_value, critic_step)
            writer.add_summary(actor_summary_value, actor_step)

            data.update(code_value)

            if (i + 1) % 100 == 0:
                hook_train = data.hook_train()
                hook_summary = tf.Summary(value=[tf.Summary.Value(tag='hook/train', simple_value=hook_train)])
                print('batch {}: actor {}, critic {}'.format(i, actor_value, critic_value))
                writer.add_summary(hook_summary, actor_step)

            if (i + 1) % 1000 == 0:
                print('Testing!!!!!!!!')
                test_batch = data.next_batch('test')
                test_dict = {self.image_in: test_batch['batch_image']}
                test_code = sess.run(self.net['codes'], feed_dict=test_dict)
                data.update(test_code, phase='test')
                hook_test = data.hook_test()
                hook_summary = tf.Summary(value=[tf.Summary.Value(tag='hook/test', simple_value=hook_test)])
                writer.add_summary(hook_summary, actor_step)

            if (i + 1) % 3000 == 0:
                self._save(sess, save_path, actor_step)

    @staticmethod
    def _restore(sess: tf.Session, restore_file, var_list=None):
        saver = tf.train.Saver(var_list=var_list)
        saver.restore(sess, save_path=restore_file)

    @staticmethod
    def _save(sess: tf.Session, save_path, step):
        saver = tf.train.Saver()
        saver.save(sess, save_path + 'YMModel', step)
        print('Saved!')


if __name__ == '__main__':
    batch_size = 200
    code_length = 32
    train_file = 'data/cifar10_vgg_fc7_train.mat'
    test_file = 'data/cifar10_vgg_fc7_test.mat'

    model_config = {'batch_size': batch_size, 'code_length': code_length}
    train_config = {'batch_size': batch_size, 'code_length': code_length, 'file_name': train_file, 'phase': 'train'}
    test_config = {'batch_size': batch_size, 'code_length': code_length, 'file_name': test_file, 'phase': 'train'}

    sess = tf.Session()

    model = BasicModel(**model_config)

    train_data = MatDataset(**train_config)
    test_data = MatDataset(**test_config)
    data_helper = DataHelper(train_data, test_data)

    model.train(sess, data_helper)
