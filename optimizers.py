# All these optimizers are from https://github.com/keras-team/keras with
# weight_decay and layer multipliers updates.

from keras.legacy import interfaces
import keras.backend as K
from keras.optimizers import Optimizer


class SGD2(Optimizer):
    """Stochastic gradient descent optimizer.
    Includes support for momentum,
    learning rate decay, and Nesterov momentum.
    # Arguments
        lr: float >= 0. Learning rate.
        momentum: float >= 0. Parameter that accelerates SGD
            in the relevant direction and dampens oscillations.
        decay: float >= 0. Learning rate decay over each update.
        nesterov: boolean. Whether to apply Nesterov momentum.
        wd: float >= 0. Weight decay over each update.
        lr_multipliers: dict {string: float}. Learning rate multipliers for
            different layers.
    """

    def __init__(self, lr=0.01, momentum=0., decay=0., nesterov=False, wd=0.,
                 lr_multipliers=None, **kwargs):
        super().__init__(**kwargs)

        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.momentum = K.variable(momentum, name='momentum')
            self.decay = K.variable(decay, name='decay')

        self.initial_decay = decay
        self.nesterov = nesterov
        self.wd = wd
        self.lr_multipliers = lr_multipliers

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr = lr * (1. / (1. + self.decay *
                             K.cast(self.iterations, K.dtype(self.decay))))
        # momentum
        shapes = [K.int_shape(p) for p in params]
        moments = [K.zeros(shape) for shape in shapes]
        self.weights = [self.iterations] + moments
        for p, g, m in zip(params, grads, moments):
            # Update lr
            new_lr = lr
            if self.lr_multipliers is not None:
                matched_layer = [
                    x for x in self.lr_multipliers.keys() if x in p.name]
                if matched_layer:
                    new_lr = lr * self.lr_multipliers[matched_layer[0]]

            v = self.momentum * m - new_lr * g
            self.updates.append(K.update(m, v))

            if self.nesterov:
                new_p = p + self.momentum * v - new_lr * g
            else:
                new_p = p + v

            # Weight_decay
            new_p -= new_lr * self.wd * p

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'momentum': float(K.get_value(self.momentum)),
                  'decay': float(K.get_value(self.decay)),
                  'nesterov': self.nesterov,
                  'wd': self.wd,
                  'lr_multipliers': self.lr_multipliers}

        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class RMSprop2(Optimizer):
    """RMSProp optimizer.
    It is recommended to leave the parameters of this optimizer
    at their default values
    (except the learning rate, which can be freely tuned).
    This optimizer is usually a good choice for recurrent
    neural networks.
    # Arguments
        lr: float >= 0. Learning rate.
        rho: float >= 0.
        epsilon: float >= 0. Fuzz factor. If `None`, defaults to `K.epsilon()`.
        decay: float >= 0. Learning rate decay over each update.
        wd: float >= 0. Weight decay over each update.
        lr_multipliers: dict {string: float}. Learning rate multipliers for
            different layers.
    # References
        - [rmsprop: Divide the gradient by a running average of its recent magnitude
           ](http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)
    """

    def __init__(self, lr=0.001, rho=0.9, epsilon=None, decay=0., wd=0.,
                 lr_multipliers=None, **kwargs):
        super().__init__(**kwargs)

        with K.name_scope(self.__class__.__name__):
            self.lr = K.variable(lr, name='lr')
            self.rho = K.variable(rho, name='rho')
            self.decay = K.variable(decay, name='decay')
            self.iterations = K.variable(0, dtype='int64', name='iterations')

        if epsilon is None:
            epsilon = K.epsilon()

        self.epsilon = epsilon
        self.initial_decay = decay
        self.wd = wd
        self.lr_multipliers = lr_multipliers

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        accumulators = [K.zeros(K.int_shape(p), dtype=K.dtype(p))
                        for p in params]
        self.weights = accumulators
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr = lr * (1. / (1. + self.decay * K.cast(self.iterations,
                                                      K.dtype(self.decay))))

        for p, g, a in zip(params, grads, accumulators):
            # Update lr
            new_lr = lr
            if self.lr_multipliers is not None:
                matched_layer = [
                    x for x in self.lr_multipliers.keys() if x in p.name]
                if matched_layer:
                    new_lr = lr * self.lr_multipliers[matched_layer[0]]

            # update accumulator
            new_a = self.rho * a + (1. - self.rho) * K.square(g)
            self.updates.append(K.update(a, new_a))
            new_p = p - new_lr * g / (K.sqrt(new_a) + self.epsilon)

            # Weight_decay
            new_p -= new_lr * self.wd * p

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'rho': float(K.get_value(self.rho)),
                  'decay': float(K.get_value(self.decay)),
                  'epsilon': self.epsilon,
                  'wd': self.wd,
                  'lr_multipliers': self.lr_multipliers}

        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Adagrad2(Optimizer):
    """Adagrad optimizer.
    Adagrad is an optimizer with parameter-specific learning rates,
    which are adapted relative to how frequently a parameter gets
    updated during training. The more updates a parameter receives,
    the smaller the updates.
    It is recommended to leave the parameters of this optimizer
    at their default values.
    # Arguments
        lr: float >= 0. Initial learning rate.
        epsilon: float >= 0. If `None`, defaults to `K.epsilon()`.
        decay: float >= 0. Learning rate decay over each update.
        wd: float >= 0. Weight decay over each update.
        lr_multipliers: dict {string: float}. Learning rate multipliers for
            different layers.
    # References
        - [Adaptive Subgradient Methods for Online Learning and Stochastic
           Optimization](http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf)
    """

    def __init__(self, lr=0.01, epsilon=None, decay=0., wd=0.,
                 lr_multipliers=None, **kwargs):
        super().__init__(**kwargs)

        with K.name_scope(self.__class__.__name__):
            self.lr = K.variable(lr, name='lr')
            self.decay = K.variable(decay, name='decay')
            self.lr_multipliers = lr_multipliers
            self.iterations = K.variable(0, dtype='int64', name='iterations')

        if epsilon is None:
            epsilon = K.epsilon()

        self.epsilon = epsilon
        self.initial_decay = decay
        self.wd = wd
        self.lr_multipliers = lr_multipliers

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        shapes = [K.int_shape(p) for p in params]
        accumulators = [K.zeros(shape) for shape in shapes]
        self.weights = accumulators
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr = lr * (1. / (1. + self.decay * K.cast(self.iterations,
                                                      K.dtype(self.decay))))

        for p, g, a in zip(params, grads, accumulators):
            # Update lr
            new_lr = lr
            if self.lr_multipliers is not None:
                matched_layer = [
                    x for x in self.lr_multipliers.keys() if x in p.name]
                if matched_layer:
                    new_lr = lr * self.lr_multipliers[matched_layer[0]]

            new_a = a + K.square(g)  # update accumulator
            self.updates.append(K.update(a, new_a))
            new_p = p - new_lr * g / (K.sqrt(new_a) + self.epsilon)

            # Weight_decay
            new_p -= new_lr * self.wd * p

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'decay': float(K.get_value(self.decay)),
                  'epsilon': self.epsilon,
                  'wd': self.wd,
                  'lr_multipliers': self.lr_multipliers}

        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Adadelta2(Optimizer):
    """Adadelta optimizer.
    Adadelta is a more robust extension of Adagrad
    that adapts learning rates based on a moving window of gradient updates,
    instead of accumulating all past gradients. This way, Adadelta continues
    learning even when many updates have been done. Compared to Adagrad, in the
    original version of Adadelta you don't have to set an initial learning
    rate. In this version, initial learning rate and decay factor can
    be set, as in most other Keras optimizers.
    It is recommended to leave the parameters of this optimizer
    at their default values.
    # Arguments
        lr: float >= 0. Initial learning rate, defaults to 1.
            It is recommended to leave it at the default value.
        rho: float >= 0. Adadelta decay factor, corresponding to fraction of
            gradient to keep at each time step.
        epsilon: float >= 0. Fuzz factor. If `None`, defaults to `K.epsilon()`.
        decay: float >= 0. Initial learning rate decay.
        wd: float >= 0. Weight decay over each update.
        lr_multipliers: dict {string: float}. Learning rate multipliers for
            different layers.
    # References
        - [Adadelta - an adaptive learning rate method](
           https://arxiv.org/abs/1212.5701)
    """

    def __init__(self, lr=1.0, rho=0.95, epsilon=None, decay=0., wd=0.,
                 lr_multipliers=None, **kwargs):
        super().__init__(**kwargs)

        with K.name_scope(self.__class__.__name__):
            self.lr = K.variable(lr, name='lr')
            self.decay = K.variable(decay, name='decay')
            self.iterations = K.variable(0, dtype='int64', name='iterations')

        if epsilon is None:
            epsilon = K.epsilon()

        self.rho = rho
        self.epsilon = epsilon
        self.initial_decay = decay
        self.wd = wd
        self.lr_multipliers = lr_multipliers

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        shapes = [K.int_shape(p) for p in params]
        accumulators = [K.zeros(shape) for shape in shapes]
        delta_accumulators = [K.zeros(shape) for shape in shapes]
        self.weights = accumulators + delta_accumulators
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr = lr * (1. / (1. + self.decay * K.cast(self.iterations,
                                                      K.dtype(self.decay))))

        for p, g, a, d_a in zip(params, grads, accumulators, delta_accumulators):
            # Update lr
            new_lr = lr
            if self.lr_multipliers is not None:
                matched_layer = [
                    x for x in self.lr_multipliers.keys() if x in p.name]
                if matched_layer:
                    new_lr = lr * self.lr_multipliers[matched_layer[0]]

            # update accumulator
            new_a = self.rho * a + (1. - self.rho) * K.square(g)
            self.updates.append(K.update(a, new_a))

            # use the new accumulator and the *old* delta_accumulator
            update = g * K.sqrt(d_a + self.epsilon) / \
                K.sqrt(new_a + self.epsilon)
            new_p = p - new_lr * update

            # Weight_decay
            new_p -= new_lr * self.wd * p

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))

            # update delta_accumulator
            new_d_a = self.rho * d_a + (1 - self.rho) * K.square(update)
            self.updates.append(K.update(d_a, new_d_a))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'rho': self.rho,
                  'decay': float(K.get_value(self.decay)),
                  'epsilon': self.epsilon,
                  'wd': self.wd,
                  'lr_multipliers': self.lr_multipliers}

        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Adam2(Optimizer):
    """Adam optimizer.
    Default parameters follow those provided in the original paper.
    # Arguments
        lr: float >= 0. Learning rate.
        beta_1: float, 0 < beta < 1. Generally close to 1.
        beta_2: float, 0 < beta < 1. Generally close to 1.
        epsilon: float >= 0. Fuzz factor. If `None`, defaults to `K.epsilon()`.
        decay: float >= 0. Learning rate decay over each update.
        amsgrad: boolean. Whether to apply the AMSGrad variant of this
            algorithm from the paper "On the Convergence of Adam and
            Beyond".
        wd: float >= 0. Weight decay over each update.
        lr_multipliers: dict {string: float}. Learning rate multipliers for
            different layers.
    # References
        - [Adam - A Method for Stochastic Optimization](
           https://arxiv.org/abs/1412.6980v8)
        - [On the Convergence of Adam and Beyond](
           https://openreview.net/forum?id=ryQu7f-RZ)
    """

    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None,
                 decay=0., amsgrad=False, wd=0., lr_multipliers=None, **kwargs):
        super().__init__(**kwargs)

        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.beta_1 = K.variable(beta_1, name='beta_1')
            self.beta_2 = K.variable(beta_2, name='beta_2')
            self.decay = K.variable(decay, name='decay')

        if epsilon is None:
            epsilon = K.epsilon()

        self.epsilon = epsilon
        self.initial_decay = decay
        self.amsgrad = amsgrad
        self.wd = wd
        self.lr_multipliers = lr_multipliers

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr = lr * (1. / (1. + self.decay * K.cast(self.iterations,
                                                      K.dtype(self.decay))))

        t = K.cast(self.iterations, K.floatx()) + 1
        lr_t = lr * (K.sqrt(1. - K.pow(self.beta_2, t)) /
                     (1. - K.pow(self.beta_1, t)))

        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        if self.amsgrad:
            vhats = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        else:
            vhats = [K.zeros(1) for _ in params]
        self.weights = [self.iterations] + ms + vs + vhats

        for p, g, m, v, vhat in zip(params, grads, ms, vs, vhats):
            # Update lr
            new_lr_t = lr_t
            if self.lr_multipliers is not None:
                matched_layer = [
                    x for x in self.lr_multipliers.keys() if x in p.name]
                if matched_layer:
                    new_lr_t = lr_t * self.lr_multipliers[matched_layer[0]]

            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)

            if self.amsgrad:
                vhat_t = K.maximum(vhat, v_t)
                p_t = p - new_lr_t * m_t / (K.sqrt(vhat_t) + self.epsilon)
                self.updates.append(K.update(vhat, vhat_t))
            else:
                p_t = p - new_lr_t * m_t / (K.sqrt(v_t) + self.epsilon)

            self.updates.append(K.update(m, m_t))
            self.updates.append(K.update(v, v_t))
            new_p = p_t

            # Weight_decay
            new_p -= new_lr_t * self.wd * p

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'beta_1': float(K.get_value(self.beta_1)),
                  'beta_2': float(K.get_value(self.beta_2)),
                  'decay': float(K.get_value(self.decay)),
                  'epsilon': self.epsilon,
                  'amsgrad': self.amsgrad,
                  'wd': self.wd,
                  'lr_multipliers': self.lr_multipliers}

        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Adamax2(Optimizer):
    """Adamax optimizer from Adam paper's Section 7.
    It is a variant of Adam based on the infinity norm.
    Default parameters follow those provided in the paper.
    # Arguments
        lr: float >= 0. Learning rate.
        beta_1/beta_2: floats, 0 < beta < 1. Generally close to 1.
        epsilon: float >= 0. Fuzz factor. If `None`, defaults to `K.epsilon()`.
        decay: float >= 0. Learning rate decay over each update.
        wd: float >= 0. Weight decay over each update.
        lr_multipliers: dict {string: float}. Learning rate multipliers for
            different layers.
    # References
        - [Adam - A Method for Stochastic Optimization](
           https://arxiv.org/abs/1412.6980v8)
    """

    def __init__(self, lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None,
                 decay=0., wd=0., lr_multipliers=None, **kwargs):
        super().__init__(**kwargs)

        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.beta_1 = K.variable(beta_1, name='beta_1')
            self.beta_2 = K.variable(beta_2, name='beta_2')
            self.decay = K.variable(decay, name='decay')

        if epsilon is None:
            epsilon = K.epsilon()

        self.epsilon = epsilon
        self.initial_decay = decay
        self.wd = wd
        self.lr_multipliers = lr_multipliers

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr = lr * (1. / (1. + self.decay * K.cast(self.iterations,
                                                      K.dtype(self.decay))))

        t = K.cast(self.iterations, K.floatx()) + 1
        lr_t = lr / (1. - K.pow(self.beta_1, t))

        shapes = [K.int_shape(p) for p in params]
        # zero init of 1st moment
        ms = [K.zeros(shape) for shape in shapes]
        # zero init of exponentially weighted infinity norm
        us = [K.zeros(shape) for shape in shapes]
        self.weights = [self.iterations] + ms + us

        for p, g, m, u in zip(params, grads, ms, us):
            # Update lr
            new_lr_t = lr_t
            if self.lr_multipliers is not None:
                matched_layer = [
                    x for x in self.lr_multipliers.keys() if x in p.name]
                if matched_layer:
                    new_lr_t = lr_t * self.lr_multipliers[matched_layer[0]]

            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            u_t = K.maximum(self.beta_2 * u, K.abs(g))
            p_t = p - new_lr_t * m_t / (u_t + self.epsilon)

            self.updates.append(K.update(m, m_t))
            self.updates.append(K.update(u, u_t))
            new_p = p_t

            # Weight_decay
            new_p -= new_lr_t * self.wd * p

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'beta_1': float(K.get_value(self.beta_1)),
                  'beta_2': float(K.get_value(self.beta_2)),
                  'decay': float(K.get_value(self.decay)),
                  'epsilon': self.epsilon,
                  'wd': self.wd,
                  'lr_multipliers': self.lr_multipliers}

        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Nadam2(Optimizer):
    """Nesterov Adam optimizer.
    Much like Adam is essentially RMSprop with momentum,
    Nadam is Adam RMSprop with Nesterov momentum.
    Default parameters follow those provided in the paper.
    It is recommended to leave the parameters of this optimizer
    at their default values.
    # Arguments
        lr: float >= 0. Learning rate.
        beta_1/beta_2: floats, 0 < beta < 1. Generally close to 1.
        epsilon: float >= 0. Fuzz factor. If `None`, defaults to `K.epsilon()`.
        wd: float >= 0. Weight decay over each update.
        lr_multipliers: dict {string: float}. Learning rate multipliers for
            different layers.
    # References
        - [Nadam report](http://cs229.stanford.edu/proj2015/054_report.pdf)
        - [On the importance of initialization and momentum in deep learning](
           http://www.cs.toronto.edu/~fritz/absps/momentum.pdf)
    """

    def __init__(self, lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None,
                 schedule_decay=0.004, wd=0., lr_multipliers=None, **kwargs):
        super().__init__(**kwargs)

        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.m_schedule = K.variable(1., name='m_schedule')
            self.lr = K.variable(lr, name='lr')
            self.beta_1 = K.variable(beta_1, name='beta_1')
            self.beta_2 = K.variable(beta_2, name='beta_2')

        if epsilon is None:
            epsilon = K.epsilon()

        self.epsilon = epsilon
        self.schedule_decay = schedule_decay
        self.wd = wd
        self.lr_multipliers = lr_multipliers

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        t = K.cast(self.iterations, K.floatx()) + 1

        # Due to the recommendations in [2], i.e. warming momentum schedule
        momentum_cache_t = self.beta_1 * (1. - 0.5 * (
            K.pow(K.cast_to_floatx(0.96), t * self.schedule_decay)))
        momentum_cache_t_1 = self.beta_1 * (1. - 0.5 * (
            K.pow(K.cast_to_floatx(0.96), (t + 1) * self.schedule_decay)))
        m_schedule_new = self.m_schedule * momentum_cache_t
        m_schedule_next = self.m_schedule * momentum_cache_t * momentum_cache_t_1
        self.updates.append((self.m_schedule, m_schedule_new))

        shapes = [K.int_shape(p) for p in params]
        ms = [K.zeros(shape) for shape in shapes]
        vs = [K.zeros(shape) for shape in shapes]

        self.weights = [self.iterations] + ms + vs

        for p, g, m, v in zip(params, grads, ms, vs):
            # Update lr
            new_lr = lr
            if self.lr_multipliers is not None:
                matched_layer = [
                    x for x in self.lr_multipliers.keys() if x in p.name]
                if matched_layer:
                    new_lr = lr * self.lr_multipliers[matched_layer[0]]

            # the following equations given in [1]
            g_prime = g / (1. - m_schedule_new)
            m_t = self.beta_1 * m + (1. - self.beta_1) * g
            m_t_prime = m_t / (1. - m_schedule_next)
            v_t = self.beta_2 * v + (1. - self.beta_2) * K.square(g)
            v_t_prime = v_t / (1. - K.pow(self.beta_2, t))
            m_t_bar = (1. - momentum_cache_t) * g_prime + (
                momentum_cache_t_1 * m_t_prime)

            self.updates.append(K.update(m, m_t))
            self.updates.append(K.update(v, v_t))

            p_t = p - new_lr * m_t_bar / (K.sqrt(v_t_prime) + self.epsilon)
            new_p = p_t

            # Weight_decay
            new_p -= new_lr * self.wd * p

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'beta_1': float(K.get_value(self.beta_1)),
                  'beta_2': float(K.get_value(self.beta_2)),
                  'epsilon': self.epsilon,
                  'schedule_decay': self.schedule_decay,
                  'wd': self.wd,
                  'lr_multipliers': self.lr_multipliers}

        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
