from model import basic_model as bm
import tensorflow as tf
from util.layer import conventional_layers as layers
from util.data.dataset import DataHelper
from time import gmtime, strftime
import os
from util.layer import graph_conv as gcn


class SGH(bm.BasicModel):
    def _build_net(self):
        with tf.variable_scope('actor'):
            with tf.variable_scope('encoder'):
                fc_1 = layers.fc_relu_layer('fc_1', bottom=self.image_in, output_dim=2048)
                # continuous_hidden = tf.nn.sigmoid(layers.fc_layer('cont_hidden', bottom=fc_1, output_dim=256))

                _batch_size, _feature_size = self.image_in.get_shape().as_list()

                eps = tf.ones([_batch_size, self.code_length], dtype=bm.D_TYPE) * 0.5
                code_hidden = layers.fc_layer('code_hidden', bottom=fc_1, output_dim=self.code_length)

                codes, code_prob = bm.doubly_sn(code_hidden, eps)

            with tf.variable_scope('decoder'):
                fc_2 = tf.nn.relu(bm.fc_layer_hack('fc_2', bottom=codes, input_dim=self.code_length, output_dim=2048),
                                  name='critic_sigmoid_2')
                decode_result = layers.fc_relu_layer('decode_result', fc_2, _feature_size)

        tf.summary.scalar('actor/binary', tf.reduce_mean(codes))

        return {
            'codes': codes,
            'code_hidden': code_hidden,
            'code_prob': code_prob,
            'decode_result': decode_result}

    def _build_loss(self):
        a = tf.nn.l2_loss(self.net.get('decode_result') - self.image_in)
        b = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.net.get('code_prob'),
                                                                   labels=self.net.get('codes')))
        tf.summary.scalar('actor/recon', a)
        tf.summary.scalar('actor/regu', b)

        return a + b

    def train(self, sess: tf.Session, data: DataHelper, restore_file=None, log_path='data'):
        actor_loss = self._build_loss()
        actor_opt = self.opt(actor_loss, 'actor')
        initial_op = tf.global_variables_initializer()

        sess.run(initial_op)

        time_string = strftime("%a%d%b%Y-%H%M%S", gmtime())
        summary_path = os.path.join(log_path, 'log', time_string) + os.sep
        save_path = os.path.join(log_path, 'model') + os.sep

        if not os.path.exists(save_path):
            os.mkdir(save_path)

        if restore_file is not None:
            self._restore(sess, restore_file)

        writer = tf.summary.FileWriter(summary_path)

        actor_summary = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES, scope='actor'))

        for i in range(70000):
            train_batch = data.next_batch('train')
            train_dict = {self.image_in: train_batch['batch_image']}
            _, actor_value, actor_summary_value, code_value, actor_step = sess.run(
                [actor_opt, actor_loss, actor_summary, self.net['codes'], self.global_step], feed_dict=train_dict)

            writer.add_summary(actor_summary_value, actor_step)

            data.update(code_value)

            if (i + 1) % 100 == 0:
                hook_train = data.hook_train()
                hook_summary = tf.Summary(value=[tf.Summary.Value(tag='hook/train', simple_value=hook_train)])
                print('batch {}: actor {}'.format(i, actor_value))
                writer.add_summary(hook_summary, actor_step)

            if (i + 1) % 5000 == 0:
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
        # data.save('', self.code_length, folder=log_path)


class SGHGCN(SGH):
    def _build_net(self):
        with tf.variable_scope('actor'):
            with tf.variable_scope('encoder'):
                fc_1 = layers.fc_relu_layer('fc_1', bottom=self.image_in, output_dim=2048)
                # continuous_hidden = tf.nn.sigmoid(layers.fc_layer('cont_hidden', bottom=fc_1, output_dim=256))

                _batch_size, _feature_size = self.image_in.get_shape().as_list()

                eps = tf.ones([_batch_size, self.code_length], dtype=bm.D_TYPE) * 0.5
                code_hidden = layers.fc_layer('code_hidden', bottom=fc_1, output_dim=self.code_length)

                codes, code_prob = bm.doubly_sn(code_hidden, eps)

                batch_adjacency = gcn.build_adjacency_hamming(codes, code_length=self.code_length)

                continuous_hidden = gcn.graph_laplacian(batch_adjacency, self.batch_size) @ codes

            with tf.variable_scope('decoder'):
                fc_2 = tf.nn.relu(
                    bm.fc_layer_hack('fc_2', bottom=continuous_hidden, input_dim=self.code_length, output_dim=2048),
                    name='critic_sigmoid_2')
                decode_result = layers.fc_relu_layer('decode_result', fc_2, _feature_size)

        tf.summary.scalar('actor/binary', tf.reduce_mean(codes))

        return {
            'codes': codes,
            'code_hidden': code_hidden,
            'code_prob': code_prob,
            'decode_result': decode_result}


class NoRegu(bm.BasicModel):
    def train(self, sess: tf.Session, data: DataHelper, restore_file=None, log_path='data'):
        actor_loss, critic_loss = self._build_loss()
        actor_opt = self.opt(actor_loss, 'actor')
        critic_opt = self.opt(critic_loss, 'critic')
        initial_op = tf.global_variables_initializer()

        sess.run(initial_op)

        time_string = strftime("%a%d%b%Y-%H%M%S", gmtime())
        summary_path = os.path.join(log_path, 'log', time_string) + os.sep
        save_path = os.path.join(log_path, 'model') + os.sep

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
            critic_value, critic_summary_value, critic_step = sess.run(
                [critic_loss, critic_summary, self.global_step], feed_dict=train_dict)
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

            if (i + 1) % 3000 == 0:
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


if __name__ == '__main__':
    from util.data.dataset import MatDataset

    batch_size = 200
    code_length = 8
    train_file = 'data/cifar10_vgg_fc7_train.mat'
    test_file = 'data/cifar10_vgg_fc7_test.mat'

    model_config = {'batch_size': batch_size, 'code_length': code_length}
    train_config = {'batch_size': batch_size, 'code_length': code_length, 'file_name': train_file, 'phase': 'train'}
    test_config = {'batch_size': batch_size, 'code_length': code_length, 'file_name': test_file, 'phase': 'train'}

    this_sess = tf.Session()

    model = SGH(**model_config)

    train_data = MatDataset(**train_config)
    test_data = MatDataset(**test_config)
    data_helper = DataHelper(train_data, test_data)

    model.train(this_sess, data_helper)
    # model.extract(this_sess, data_helper, log_path='data', task='cifar')
