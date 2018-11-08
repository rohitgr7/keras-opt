from tensorflow.keras.callbacks import Callback
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import numpy as np


class CyclicLRScheduler(Callback):
    def __init__(self, start_lr=1e-3, end_lr=6e-3, step_size=1000., decay=None, gamma=1.):
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.step_size = step_size
        self.decay = decay
        self.gamma = gamma
        self.history = {}
        self.iterations = 0

    def on_train_begin(self, _):
        K.set_value(self.model.optimizer.lr, self.cal_lr())

    def on_batch_end(self, _, logs={}):
        self.iterations += 1
        self.history.setdefault('lrs', []).append(K.get_value(self.model.optimizer.lr))
        K.set_value(self.model.optimizer.lr, self.cal_lr())

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

    def cal_lr(self):
        cycle = np.floor(1 + self.iterations / (2 * self.step_size))
        x = np.abs((self.iterations / self.step_size) - (2 * cycle) + 1)
        new_lr = self.start_lr + (self.end_lr - self.start_lr) * np.maximum(0, 1 - x)

        if self.decay == 'fixed':
            new_lr /= (2. ** (cycle - 1))
        elif self.decay == 'exp':
            new_lr *= self.gamma ** self.iterations

        return new_lr

    def plot_lr(self):
        plt.plot(range(len(self.history['lrs'])), self.history['lrs'])
        plt.xlabel('Iterations')
        plt.ylabel('Learning rate')


class SGDRScheduler(Callback):
    def __init__(self, start_lr=1e2, end_lr=1e-2, lr_decay=1., cycle_len=1, cycle_mult=1, steps_per_epoch=10):
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.lr_decay = lr_decay
        self.cycle_len = cycle_len
        self.cycle_mult = cycle_mult
        self.steps_per_epoch = steps_per_epoch
        self.batch_cycle = 0
        self.history = {}

    def on_train_begin(self, _):
        K.set_value(self.model.optimizer.lr, self.start_lr)

    def on_batch_end(self, _, logs={}):
        self.batch_cycle += 1
        self.history.setdefault('lrs', []).append(K.get_value(self.model.optimizer.lr))
        K.set_value(self.model.optimizer.lr, self.cal_lr())

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

    def on_train_end(self, _):
        self.batch_cycle = 0
        self.cycle_len = np.ceil(self.cycle_len * self.cycle_mult)
        self.start_lr *= self.lr_decay

    def cal_lr(self):
        pct = self.batch_cycle / (self.steps_per_epoch * self.cycle_len)
        cos_out = 1 + np.cos(np.pi * pct)

        return self.end_lr + 0.5 * (self.start_lr - self.end_lr) * cos_out

    def plot_lr(self):
        plt.plot(range(len(self.history['lrs'])), self.history['lrs'])
        plt.xlabel('Iterations')
        plt.ylabel('Learning rate')


def fit_cycle(model, X_train, y_train, validation_data=None, batch_size=64, epochs=1, cycle_len=1, cycle_mult=1, lr_sched=None, callbacks=None):
    cb_list = []
    cb_list = cb_list + [lr_sched] if lr_sched is not None else cb_list
    cb_list = cb_list + callbacks if callbacks is not None else cb_list

    for _ in range(epochs):
        model.fit(X_train, y_train, epochs=cycle_len, batch_size=batch_size, validation_data=validation_data, callbacks=cb_list)
        cycle_len *= cycle_mult
