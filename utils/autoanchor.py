# Auto-anchor utils

import numpy as np
import yaml
from scipy.cluster.vq import kmeans
from tqdm import tqdm

from mindspore import nn, ops, Tensor

def check_anchor_order(m):
    # Check anchor order against stride order for YOLO Detect() module m, and correct if necessary
    a = ops.ReduceProd()(m.anchor_grid, -1).view(-1) # anchor area
    da = a[-1] - a[0]  # delta a
    ds = m.stride[-1] - m.stride[0]  # delta s
    if ops.Sign()(da) != ops.Sign()(ds): # same order
        print('Reversing anchor order')
        m.anchors[:] = ops.ReverseV2(axis=0)(m.anchors)
        m.anchor_grid[:] = ops.ReverseV2(axis=0)(m.anchor_grid)