from mindspore import nn, ops

class ResizeNearestNeighbor(nn.Cell):
    def __init__(self, scale=2):
        super(ResizeNearestNeighbor, self).__init__()
        self.scale = scale
    def construct(self, x):
        return ops.ResizeNearestNeighbor((x.shape[-2] * 2, x.shape[-1] * 2))(x)
