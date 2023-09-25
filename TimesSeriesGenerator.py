from keras.utils import Sequence
import numpy as np


class TimeSeriesGenerator(Sequence):
    def __init__(self, data, targets, length, predict, mode):
        self.data = data
        self.targets = targets
        self.length = length
        self.predict = predict
        self.mode = mode

    def __len__(self):
        if self.mode == 'train':
            return ((len(self.data) - self.length) // 1) - (self.predict - 1)
        elif self.mode == 'test':
            return (len(self.data) - self.length) // self.predict
        else:
            raise ValueError("Mode must be 'train' or 'test'")

    def __getitem__(self, idx):
        if self.mode == 'train':
            data_idx = np.arange(idx, (idx + 1), self.predict)
        elif self.mode == 'test':
            data_idx = np.arange(idx * self.predict, (idx + 1) * self.predict, self.predict)
        else:
            raise ValueError("Mode must be 'train' or 'test'")

        data = np.array([self.data[i: i + self.length] for i in data_idx])
        targets = np.array([self.targets[i + self.length: i + self.length + self.predict] for i in data_idx])

        return data, targets
