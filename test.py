from model.basic_model import BasicModel
from util.data.dataset import H5Dataset, DataHelper
import tensorflow as tf

batch_size = 400
code_length = 64
train_file = '../../Data/imagenet_vgg_fc7_database.h5'
test_file = '../../Data/imagenet_vgg_fc7_test.h5'

model_config = {'batch_size': batch_size, 'code_length': code_length}
train_config = {'batch_size': batch_size, 'code_length': code_length, 'file_name': train_file, 'phase': 'train'}
test_config = {'batch_size': batch_size, 'code_length': code_length, 'file_name': test_file, 'phase': 'train'}

sess = tf.Session()

model = BasicModel(**model_config)

train_data = H5Dataset(**train_config)
test_data = H5Dataset(**test_config)
data_helper = DataHelper(train_data, test_data)

model.train(sess, data_helper, log_path='../../Log', task='imagenet')
