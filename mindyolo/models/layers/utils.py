import math
import numpy as np
import mindspore as ms
from mindspore import nn, ops, Tensor


def make_divisible(x, divisor):
    # Returns x evenly divisible by divisor
    return math.ceil(x / divisor) * divisor

def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

def calculate_fan_in_and_fan_out(shape):
    dimensions = len(shape)
    if dimensions < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")

    num_input_fmaps = shape[1]
    num_output_fmaps = shape[0]
    receptive_field_size = 1
    if dimensions > 2:
        # math.prod is not always available, accumulate the product manually
        # we could use functools.reduce but that is not supported by TorchScript
        for s in shape[2:]:
            receptive_field_size *= s
    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out

def initialize_weights(model):
    for n, m in model.cells_and_names():
        if isinstance(m, nn.Conv2d):
            pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
            pass
            # This modification is invalid.
            # m.eps = 1e-3
            # m.momentum = 0.03

def init_bias(conv_weight_shape):
    bias_init = None
    fan_in, _ = calculate_fan_in_and_fan_out(conv_weight_shape)
    if fan_in != 0:
        bound = 1 / math.sqrt(fan_in)
        bias_init = Tensor(np.random.uniform(-bound, bound, conv_weight_shape[0]), dtype=ms.float32)
    return bias_init

def check_anchor_order(m):
    # Check anchor order against stride order for YOLO Detect() module m, and correct if necessary
    a = ops.ReduceProd()(m.anchor_grid, -1).view(-1) # anchor area
    da = a[-1] - a[0]  # delta a
    ds = m.stride[-1] - m.stride[0]  # delta s
    if ops.Sign()(da) != ops.Sign()(ds): # same order
        print('Reversing anchor order')
        m.anchors[:] = ops.ReverseV2(axis=0)(m.anchors)
        m.anchor_grid[:] = ops.ReverseV2(axis=0)(m.anchor_grid)