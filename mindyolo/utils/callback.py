import os
import mindspore as ms
from mindspore import ops, Tensor, Callback
from mindspore.dataset import WaitedDSCallback

class EvalCallback(Callback):
    pass