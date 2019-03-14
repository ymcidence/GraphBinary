import numpy as np
import scipy.io as sio
from util.eval_tools import eval_cls_map
import h5py


class BasicDataset(object):
    def __init__(self, **kwargs):
        self.label_size = kwargs.get('label_size')
        self.batch_size = kwargs.get('batch_size', 100)
        self.code_length = kwargs.get('code_length', 32)
        self.phase = kwargs.get('phase', 'train')
        self.data = np.asarray([0])
        self.label = np.asarray([0])
        self.code = np.zeros((self.set_size, self.code_length))
        self.batch_count = 0
        self.this_batch = dict()

    @property
    def set_size(self):
        return self.data.shape[0]

    @property
    def batch_num(self):
        return self.set_size // self.batch_size

    def _shuffle(self):
        index = np.arange(self.set_size)
        np.random.shuffle(index)
        self.data = self.data[index, ...]
        self.label = self.label[index, ...]
        self.code = self.code[index, ...]

    def next_batch(self):
        pass

    def update(self, code: np.ndarray):
        batch_start = self.this_batch.get('batch_start')
        batch_end = batch_start + self.batch_size
        self.code[batch_start:batch_end, ...] = code
        self.this_batch['batch_code'] = code


class MatDataset(BasicDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.file_name = kwargs.get('file_name')
        self._load_data()
        self.code = np.zeros((self.set_size, self.code_length))

    # noinspection PyUnboundLocalVariable
    def _load_data(self):
        mat_file = sio.loadmat(self.file_name)
        for i in mat_file.keys():
            if i.find('label') >= 0:
                label_key = i
            if i.find('data') >= 0:
                data_key = i
        self.data = np.asarray(mat_file[data_key], dtype=np.float32)
        sparse_label = np.asarray(mat_file[label_key], dtype=np.int32)
        real_label = np.zeros((self.set_size, np.max(sparse_label) + 1))
        for i in range(self.set_size):
            real_label[i, sparse_label[i, 0]] = 1
        self.label = real_label
        del mat_file

    def next_batch(self):
        if self.batch_count == 0:
            self._shuffle()

        batch_start = self.batch_count * self.batch_size
        batch_end = batch_start + self.batch_size

        batch_image = self.data[batch_start:batch_end, ...]
        batch_label = self.label[batch_start:batch_end, ...]

        self.batch_count = (self.batch_count + 1) % self.batch_num

        self.this_batch['batch_image'] = batch_image
        self.this_batch['batch_label'] = batch_label
        self.this_batch['batch_start'] = batch_start
        self.this_batch['batch_end'] = batch_end

        return self.this_batch


class H5Dataset(MatDataset):
    def _load_data(self):
        this_file = h5py.File(self.file_name, 'r')
        self.data = this_file['data'][:].squeeze()
        self.label = this_file['label'][:].squeeze()
        this_file.close()


class DataHelper(object):
    def __init__(self, training_data: BasicDataset, test_data: BasicDataset):
        self.training_data = training_data
        self.test_data = test_data

    def next_batch(self, phase='train'):
        return self.training_data.next_batch() if phase == 'train' else self.test_data.next_batch()

    def update(self, code: np.ndarray, phase='train'):
        return self.training_data.update(code) if phase == 'train' else self.test_data.update(code)

    def hook_train(self):
        q = self.training_data.this_batch['batch_code']
        l = self.training_data.this_batch['batch_label']

        return eval_cls_map(q, q, l, l)

    def hook_test(self):
        q = self.test_data.this_batch['batch_code']
        ql = self.test_data.this_batch['batch_label']

        t = self.training_data.code
        tl = self.training_data.label
        return eval_cls_map(q, t, ql, tl, at=1000)

    def save(self, set_name, length, folder='data'):
        to_save = {'set_code': self.training_data.code,
                   'set_label': self.training_data.label,
                   'test_code': self.test_data.code,
                   'test_label': self.test_data.label}
        sio.savemat('{}/code/{}_{}.mat'.format(folder, set_name, length), to_save)


if __name__ == '__main__':
    file_name = '/home/ymcidence/Workspace/Data/vgg_feature/imagenet_vgg_fc7_test.h5'
    conf = {'batch_size': 200, 'code_length': 5, 'file_name': file_name, 'phase': 'train'}
    dataset = H5Dataset(**conf)
    print('hehe')
