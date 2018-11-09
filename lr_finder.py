from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import math


class LRFinder:

    def __init__(self, model):
        self.model = model
        self.lrs = []
        self.losses = []
        self.best_loss = 1e9

    def on_batch_end(self, batch, logs=None):
        # Get and store the learning rate
        lr = K.get_value(self.model.optimizer.lr)
        curr_loss = logs['loss']
        self.lrs.append(lr)
        self.losses.append(curr_loss)

        # Store the lowest score
        if curr_loss < self.best_loss:
            self.best_loss = curr_loss

        # Check the conditions
        if (curr_loss > 4 * self.best_loss) or math.isnan(curr_loss):
            self.model.stop_training = True
            return

        # Update lr
        lr *= self.lr_multi
        K.set_value(self.model.optimizer.lr, lr)

    def find(self, X, y, batch_size=32, epochs=100, start_lr=1e-3, end_lr=10):
        # batches and lr_multiplier
        nb = X.shape[0] // batch_size
        self.lr_multi = (float(end_lr) / start_lr) ** (1. / nb)

        # Get original weights and lr
        org_lr = K.get_value(self.model.optimizer.lr)
        org_weights = self.model.get_weights()

        # Train with callback
        K.set_value(self.model.optimizer.lr, start_lr)
        callback = LambdaCallback(
            on_batch_end=lambda batch, logs: self.on_batch_end(batch, logs))
        self.model.fit(X, y, batch_size=batch_size,
                       epochs=epochs, callbacks=[callback])

        # Set original weights and le
        K.set_value(self.model.optimizer.lr, org_lr)
        self.model.set_weights(org_weights)

    def plot_lr(self):
        plt.plot(range(len(self.lrs)), self.lrs)
        plt.xlabel('Iterations')
        plt.ylabel('Learning rate')

    def plot_loss(self, n_skip_start=10, n_skip_end=5):
        plt.plot(self.lrs[n_skip_start:-(n_skip_end + 1)],
                 self.losses[n_skip_start:-(n_skip_end + 1)])
        plt.xlabel('Learning rate (log scale)')
        plt.ylabel('Loss')
        plt.xscale('log')

    def plot_dloss(self, sma=1, n_skip_start=10, n_skip_end=5,
                   y_lim=(-0.01, 0.01)):
        derivatives = [0.] * (sma)

        for i in range(sma, len(self.lrs)):
            derivative = (self.losses[i] - self.losses[i - sma]) / sma
            derivatives.append(derivative)

        plt.plot(self.lrs[n_skip_start:-(n_skip_end + 1)],
                 derivatives[n_skip_start:-(n_skip_end + 1)])
        plt.xlabel("Learning rate (log scale)")
        plt.ylabel("dLoss")
        plt.xscale('log')
        plt.ylim(y_lim)
