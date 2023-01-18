import numpy as np
import mindspore as ms
from mindspore import nn, Tensor

class ImplicitA(nn.Cell):
    def __init__(self, channel, mean=0., std=.02):
        super(ImplicitA, self).__init__()
        self.channel = channel
        self.mean = mean
        self.std = std
        self.implicit = ms.Parameter(Tensor(np.random.normal(self.mean, self.std, (1, channel, 1, 1)), ms.float32))

    def construct(self, x):
        return self.implicit + x

class ImplicitM(nn.Cell):
    def __init__(self, channel, mean=0., std=.02):
        super(ImplicitM, self).__init__()
        self.channel = channel
        self.mean = mean
        self.std = std
        self.implicit = ms.Parameter(Tensor(np.random.normal(self.mean, self.std, (1, channel, 1, 1)), ms.float32))

    def construct(self, x):
        return self.implicit * x
