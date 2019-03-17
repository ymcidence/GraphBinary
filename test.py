from model.baseline_model import SGH as BasicModel
from util.data.dataset import H5Dataset, DataHelper
import tensorflow as tf

task = 'imagenet'
batch_size = 600
code_length = 64
train_file = '../../Data/{}_vgg_fc7_train.h5'.format(task)
test_file = '../../Data/{}_vgg_fc7_test.h5'.format(task)
base_file = '../../Data/{}_vgg_fc7_database.h5'.format(task)

model_config = {'batch_size': batch_size, 'code_length': code_length}
train_config = {'batch_size': batch_size, 'code_length': code_length, 'file_name': train_file, 'phase': 'train'}
test_config = {'batch_size': batch_size, 'code_length': code_length, 'file_name': test_file, 'phase': 'train'}
base_config = {'batch_size': batch_size, 'code_length': code_length, 'file_name': base_file, 'phase': 'train'}

sess = tf.Session()

model = BasicModel(**model_config)

train_data = H5Dataset(**train_config)
test_data = H5Dataset(**test_config)
test_data1 = H5Dataset(**test_config)
base_data = H5Dataset(**base_config)
data_helper = DataHelper(train_data, test_data)
base_helper = DataHelper(base_data, test_data1)

model.train(sess, data_helper, log_path='../../Log')
model.extract(sess, base_helper, log_path='../../Log', task=task)
