# Keras OPT Helpers
Contains keras callbacks and optimizers for training keras models

## LR Finder
### Finds optimal learning rate for model - [paper](https://arxiv.org/pdf/1506.01186.pdf) (section 3.3)


## Schedulers
Contain following schedulers:
* SGD with warm restarts - [paper](https://arxiv.org/pdf/1608.03983.pdf)
* Cyclic learning rates - [paper](https://arxiv.org/pdf/1506.01186.pdf)


## Optimizers
Contains optimizers from official keras repo added with some optimization techniques.
* Weight decay
* Discriminative learning rates


## TODO
* Weight decay normalization with wd_multi(below algo 2) and adam with restarts - [paper](https://arxiv.org/pdf/1711.05101.pdf)

