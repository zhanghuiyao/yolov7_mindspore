import math
from mindspore import nn, ops
from mindspore.common.initializer import HeUniform

from .identity import Identity
from .utils import autopad, init_bias

class Conv(nn.Cell):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True, sync_bn=False):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s,
                              pad_mode="pad",
                              padding=autopad(k, p),
                              group=g,
                              has_bias=False,
                              weight_init=HeUniform(negative_slope=math.sqrt(5)))

        if sync_bn:
            self.bn = nn.SyncBatchNorm(c2, momentum=(1 - 0.03), eps=1e-3)
        else:
            self.bn = nn.BatchNorm2d(c2, momentum=(1 - 0.03), eps=1e-3)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Cell) else Identity)

    def construct(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))

class RepConv(nn.Cell):
    # Represented convolution
    # https://arxiv.org/abs/2101.03697

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, act=True, deploy=False, sync_bn=False):
        super(RepConv, self).__init__()

        self.deploy = deploy
        self.groups = g
        self.in_channels = c1
        self.out_channels = c2

        assert k == 3
        assert autopad(k, p) == 1

        padding_11 = autopad(k, p) - k // 2

        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Cell) else Identity)

        if sync_bn:
            BatchNorm = nn.SyncBatchNorm
        else:
            BatchNorm = nn.BatchNorm2d

        if deploy:
            self.rbr_reparam = nn.Conv2d(c1, c2, k, s,
                                         pad_mode="pad",
                                         padding=autopad(k, p),
                                         group=g,
                                         has_bias=True,
                                         weight_init=HeUniform(negative_slope=math.sqrt(5)),
                                         bias_init=init_bias((c2, c1 // g, k, k)))

        else:
            self.rbr_identity = BatchNorm(num_features=c1, momentum=(1 - 0.03), eps=1e-3) if c2 == c1 and s == 1 else None

            self.rbr_dense = nn.SequentialCell([
                nn.Conv2d(c1, c2, k, s,
                          pad_mode="pad",
                          padding=autopad(k, p),
                          group=g,
                          has_bias=False,
                          weight_init=HeUniform(negative_slope=math.sqrt(5))),
                BatchNorm(num_features=c2, momentum=(1 - 0.03), eps=1e-3),
            ])
            # self.rbr_dense_conv = nn.Conv2d(c1, c2, k, s,
            #                                 pad_mode="pad",
            #                                 padding=autopad(k, p),
            #                                 group=g,
            #                                 has_bias=False,
            #                                 weight_init=HeUniform(negative_slope=math.sqrt(5)))
            # self.rbr_dense_norm = BatchNorm(num_features=c2, momentum=(1 - 0.03), eps=1e-3)

            self.rbr_1x1 = nn.SequentialCell(
                nn.Conv2d(c1, c2, 1, s,
                          pad_mode="pad",
                          padding=padding_11,
                          group=g,
                          has_bias=False,
                          weight_init=HeUniform(negative_slope=math.sqrt(5))),
                BatchNorm(num_features=c2, momentum=(1 - 0.03), eps=1e-3),
            )
            # self.rbr_1x1_conv = nn.Conv2d(c1, c2, 1, s,
            #                               pad_mode="pad",
            #                               padding=padding_11,
            #                               group=g,
            #                               has_bias=False,
            #                               weight_init=HeUniform(negative_slope=math.sqrt(5)))
            # self.rbr_1x1_norm = BatchNorm(num_features=c2, momentum=(1 - 0.03), eps=1e-3)

    def construct(self, inputs):
        if self.deploy:
            return self.act(self.rbr_reparam(inputs))

        if self.rbr_identity is None:
            id_out = 0.0
        else:
            id_out = self.rbr_identity(inputs)

        return self.act(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)
        # return self.act(self.rbr_dense_norm(self.rbr_dense_conv(inputs)) + \
        #                 self.rbr_1x1_norm(self.rbr_1x1_conv(inputs)) + \
        #                 id_out)

class DownC(nn.Cell):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, c1, c2, n=1, k=2):
        super(DownC, self).__init__()
        c_ = c1  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2//2, 3, k)
        self.cv3 = Conv(c1, c2//2, 1, 1)
        self.mp = nn.MaxPool2d(kernel_size=k, stride=k)

    def construct(self, x):
        return ops.concat((self.cv2(self.cv1(x)), self.cv3(self.mp(x))), axis=1)
