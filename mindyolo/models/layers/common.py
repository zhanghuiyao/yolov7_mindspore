from mindspore import nn, ops

class Shortcut(nn.Cell):
    def __init__(self, dimension=0):
        super(Shortcut, self).__init__()
        self.d = dimension

    def construct(self, x):
        return x[0] + x[1]

class Concat(nn.Cell):
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def construct(self, x):
        return ops.concat(x, self.d)

class ReOrg(nn.Cell):
    def __init__(self):
        super(ReOrg, self).__init__()

    def construct(self, x):
        # in: (b,c,w,h) -> out: (b,4c,w/2,h/2)
        x1 = x[:, :, ::2, ::2]
        x2 = x[:, :, 1::2, ::2]
        x3 = x[:, :, ::2, 1::2]
        x4 = x[:, :, 1::2, 1::2]
        out = ops.concat((x1, x2, x3, x4), 1)
        return out
