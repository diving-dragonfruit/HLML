#  NRML-Pytorch
PyTorch implementation of Neural Routing in Meta Learning (NRML) based on [Model-Agnostic Meta-Learning (MAML)](https://arxiv.org/abs/1703.03400).

Support `Omniglot` and `MiniImageNet` datasets.

**Core idea**: Select certain CNN filters / neurons to update based on the magnitudes of the scaling factor of the batch normalization layers.

# Acknowledgement
This repo is largely built based on this excellent [PyTorch implementation of MAML](https://github.com/dragen1860/MAML-Pytorch).

# Platform
- python: 3.x
- Pytorch: 0.4+

# MiniImagenet

## Howto

1. download `MiniImagenet` dataset from [here](https://github.com/dragen1860/LearningToCompare-Pytorch/issues/4), splitting: `train/val/test.csv` from [here](https://github.com/twitter/meta-learning-lstm/tree/master/data/miniImagenet).
2. extract it like:
```shell
miniimagenet/
├── images
	├── n0210891500001298.jpg  
	├── n0287152500001298.jpg 
	...
├── test.csv
├── val.csv
└── train.csv


```
3. modify the `path` in `miniimagenet_train.py` to your actual data path:
```python
  root = 'miniimagenet/'
```

4. Run `python miniimagenet_train.py` with the specificed arguments.
   - `p_task` is the percentage list for selecting neurons in the inner/task loop.
   - `p_meta` is the percentage list for selecting neurons in the outer/meta loop.
   - Ex. a value of 1 means selecting 100% of the neurons; a value of 0.8 means selecting 80% of the neurons.
   - They are of length 4 because there are 4 CNN layers in the network.



# Ominiglot

## Howto
1. Run `python omniglot_train.py` with the specificed arguments; the program will download `omniglot` dataset automatically.

