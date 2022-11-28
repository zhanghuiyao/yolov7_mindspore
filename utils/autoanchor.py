# Auto-anchor utils

from mindspore import ops

def check_anchor_order(m):
    # Check anchor order against stride order for YOLO Detect() module m, and correct if necessary
    a = ops.ReduceProd()(m.anchor_grid, -1).view(-1) # anchor area
    da = a[-1] - a[0]  # delta a
    ds = m.stride[-1] - m.stride[0]  # delta s
    if ops.Sign()(da) != ops.Sign()(ds): # same order
        print('Reversing anchor order')
        m.anchors[:] = ops.ReverseV2(axis=0)(m.anchors)
        m.anchor_grid[:] = ops.ReverseV2(axis=0)(m.anchor_grid)
