import numpy as np
import scipy.io as sio
from util.eval_tools import eval_cls_map


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
        if self.batch_count == 0 and self.phase == 'train':
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


if __name__ == '__main__':
    config = dict(
        set_size=160859,
        batch_size=96,
        data_path='E:\\WorkSpace\\Data\\TU\\ImageResized\\',
        meta_file='E:\\WorkSpace\\Data\\TU\\Meta\\img_train.mat',
        label_size=200
    )
    this_set = BasicDataset(**config)
    this_set.next_batch_train()
    print('hehe')
