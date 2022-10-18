import math
import numpy as np
import mindspore as ms
import mindspore.numpy as mnp
from mindspore import nn, ops, Tensor
from mindspore.common.initializer import HeUniform
from utils.general import make_divisible

_SYNC_BN = False

class ResizeNearestNeighbor(nn.Cell):
    def __init__(self, scale=2):
        super(ResizeNearestNeighbor, self).__init__()
        self.scale = scale
    def construct(self, x):
        return ops.ResizeNearestNeighbor((x.shape[-2] * 2, x.shape[-1] * 2))(x)

def _calculate_fan_in_and_fan_out(shape):
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

def _init_bias(conv_weight_shape):
    bias_init = None
    fan_in, _ = _calculate_fan_in_and_fan_out(conv_weight_shape)
    if fan_in != 0:
        bound = 1 / math.sqrt(fan_in)
        bias_init = Tensor(np.random.uniform(-bound, bound, conv_weight_shape[0]), dtype=ms.float32)
    return bias_init

def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class MP(nn.Cell):
    def __init__(self, k=2):
        super(MP, self).__init__()
        self.m = nn.MaxPool2d(kernel_size=k, stride=k)

    def construct(self, x):
        return self.m(x)

class Concat(nn.Cell):
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def construct(self, x):
        return ops.concat(x, self.d)

class Conv(nn.Cell):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s,
                              pad_mode="pad",
                              padding=autopad(k, p),
                              group=g,
                              has_bias=False,
                              weight_init=HeUniform(negative_slope=5))
        if _SYNC_BN:
            self.bn = nn.SyncBatchNorm(c2)
        else:
            self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Cell) else nn.Identity())

    def construct(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))

class PoolWithPad(nn.Cell):
    def __init__(self, kernel_size, stride, padding):
        super(PoolWithPad, self).__init__()
        assert isinstance(padding, int)
        self.pad = nn.Pad(paddings=((0, 0), (0, 0), (padding, padding), (padding, padding)))
        self.pool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)

    def construct(self, x):
        x = self.pad(x)
        x = self.pool(x)
        return x

class SPPCSPC(nn.Cell):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=(5, 9, 13)):
        super(SPPCSPC, self).__init__()
        c_ = int(2 * c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(c_, c_, 3, 1)
        self.cv4 = Conv(c_, c_, 1, 1)
        self.m = nn.CellList([PoolWithPad(kernel_size=x, stride=1, padding=x // 2) for x in k])
        self.cv5 = Conv(4 * c_, c_, 1, 1)
        self.cv6 = Conv(c_, c_, 3, 1)
        self.cv7 = Conv(2 * c_, c2, 1, 1)

    def construct(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        m_tuple = (x1,)
        for i in range(len(self.m)):
            m_tuple += (self.m[i](x1),)
        y1 = self.cv6(self.cv5(ops.Concat(axis=1)(m_tuple)))
        # y1 = self.cv6(self.cv5(torch.cat([x1] + [m(x1) for m in self.m], 1)))
        y2 = self.cv2(x)
        return self.cv7(ops.Concat(axis=1)((y1, y2)))
        # return self.cv7(torch.cat((y1, y2), dim=1))


class RepConv(nn.Cell):
    # Represented convolution
    # https://arxiv.org/abs/2101.03697

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, act=True, deploy=False):
        super(RepConv, self).__init__()

        self.deploy = deploy
        self.groups = g
        self.in_channels = c1
        self.out_channels = c2

        assert k == 3
        assert autopad(k, p) == 1

        padding_11 = autopad(k, p) - k // 2

        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Cell) else nn.Identity())

        if _SYNC_BN:
            BatchNorm = nn.SyncBatchNorm
        else:
            BatchNorm = nn.BatchNorm2d

        if deploy:
            _conv_weight_shape = (c2, c1 // g, k, k)
            self.rbr_reparam = nn.Conv2d(c1, c2, k, s,
                                         pad_mode="pad",
                                         padding=autopad(k, p),
                                         group=g,
                                         has_bias=True,
                                         weight_init=HeUniform(negative_slope=5),
                                         bias_init=_init_bias(_conv_weight_shape))

        else:
            self.rbr_identity = (BatchNorm(num_features=c1) if c2 == c1 and s == 1 else None)

            # self.rbr_dense = nn.SequentialCell([
            #     nn.Conv2d(c1, c2, k, s,
            #               pad_mode="pad",
            #               padding=autopad(k, p),
            #               group=g,
            #               has_bias=False,
            #               weight_init=HeUniform(negative_slope=5)),
            #     BatchNorm(num_features=c2),
            # ])
            self.rbr_dense_conv = nn.Conv2d(c1, c2, k, s,
                                            pad_mode="pad",
                                            padding=autopad(k, p),
                                            group=g,
                                            has_bias=False,
                                            weight_init=HeUniform(negative_slope=5))
            self.rbr_dense_norm = BatchNorm(num_features=c2)

            # self.rbr_1x1 = nn.SequentialCell(
            #     nn.Conv2d(c1, c2, 1, s,
            #               pad_mode="pad",
            #               padding=padding_11,
            #               group=g,
            #               has_bias=False,
            #               weight_init=HeUniform(negative_slope=5)),
            #     BatchNorm(num_features=c2),
            # )
            self.rbr_1x1_conv = nn.Conv2d(c1, c2, 1, s,
                                          pad_mode="pad",
                                          padding=padding_11,
                                          group=g,
                                          has_bias=False,
                                          weight_init=HeUniform(negative_slope=5))
            self.rbr_1x1_norm = BatchNorm(num_features=c2)

    def construct(self, inputs):
        if self.deploy:
            return self.act(self.rbr_reparam(inputs))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.act(self.rbr_dense_norm(self.rbr_dense_conv(inputs)) + \
                        self.rbr_1x1_norm(self.rbr_1x1_conv(inputs)) + id_out)


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

class BaseCell(nn.Cell):
    def __init__(self, parameter):
        super(BaseCell, self).__init__()
        self.param = parameter

@ops.constexpr(reuse_result=True)
def get_convert_matrix():
    return Tensor(np.array([[1, 0, 1, 0], [0, 1, 0, 1], [-0.5, 0, 0.5, 0], [0, -0.5, 0, 0.5]]),
                  dtype=ms.float32)

class IDetect(nn.Cell):
    # stride = None  # strides computed during build
    # export = False  # onnx export
    # end2end = False
    # include_nms = False
    # concat = False

    def __init__(self, nc=80, anchors=(), ch=()):  # detection layer
        super(IDetect, self).__init__()
        self.stride = None
        self.export = False
        self.end2end = False
        self.include_nms = False
        self.concat = False

        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid_cell = nn.CellList([BaseCell(ms.Parameter(Tensor(np.zeros(1), ms.float32),
                                                            requires_grad=False))
                                      for _ in range(self.nl)])
        # self.grid = [Tensor(np.zeros(1), ms.float32)] * self.nl  # init grid
        self.anchors = ms.Parameter(Tensor(anchors, ms.float32).view(self.nl, -1, 2),
                                    requires_grad=False) # shape(nl,na,2)
        self.anchor_grid = ms.Parameter(Tensor(anchors, ms.float32).view(self.nl, 1, -1, 1, 1, 2),
                                        requires_grad=False) # shape(nl,1,na,1,1,2)

        self.m = nn.CellList([nn.Conv2d(x, self.no * self.na, 1,
                                        pad_mode="valid",
                                        has_bias=True,
                                        weight_init=HeUniform(negative_slope=5),
                                        bias_init=_init_bias((self.no * self.na, x, 1, 1))) for x in ch])  # output conv


        self.ia = nn.CellList([ImplicitA(x) for x in ch])
        self.im = nn.CellList([ImplicitM(self.no * self.na) for _ in ch])

    def construct(self, x):
        # x = x.copy()  # for profiling
        # z = ()  # inference output

        outs = ()
        for i in range(self.nl):
            out = self.m[i](self.ia[i](x[i]))  # conv
            out = self.im[i](out)
            bs, _, ny, nx = out.shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            out = ops.Transpose()(out.view(bs, self.na, self.no, ny, nx), (0, 1, 3, 4, 2))
            out = out
            outs += (out,)

        #     if not self.training:  # inference
        #         grid_i_shape = self.grid_cell[i].param.shape
        #         x_i_shape = x[i].shape
        #         if grid_i_shape[2] != x_i_shape[2] or grid_i_shape[3] != x_i_shape[3]:
        #         # if self.grid[i].shape[2:4] != x[i].shape[2:4]:
        #         #     self.grid_cell[i].param = self._make_grid(nx, ny, self.grid_cell[i].param.dtype)
        #             self.grid_cell[i].param = ops.assign(self.grid_cell[i].param, self._make_grid(nx, ny, self.grid_cell[i].param.dtype))
        #
        #         y = ops.Sigmoid()(x[i])
        #         y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid_cell[i].param) * self.stride[i]  # xy
        #         y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
        #         z += (y.view(bs, -1, self.no),)

        return outs
        # return outs if self.training else (ops.concat(z, 1), outs)

    def fuseforward(self, x):
        # x = x.copy()  # for profiling
        z = ()  # inference output
        self.training |= self.export
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = ops.transpose(x[i].view(bs, self.na, self.no, ny, nx), (0, 1, 3, 4, 2))
            x[i] = x[i]

            if not self.training:  # inference
                grid_i_shape = self.grid_cell[i].param.shape
                x_i_shape = x[i].shape
                if grid_i_shape[2] != x_i_shape[2] or grid_i_shape[3] != x_i_shape[3]:
                    self.grid_cell[i].param = self._make_grid(nx, ny, self.grid_cell[i].param.dtype)

                y = ops.Sigmoid()(x[i])
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid_cell[i].param) * self.stride[i]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                z += (y.view(bs, -1, self.no),)

        if self.training:
            out = x
        elif self.end2end:
            out = ops.concat(z, 1)
        elif self.include_nms:
            z = self.convert(z)
            out = (z,)
        elif self.concat:
            out = ops.concat(z, 1)
        else:
            out = (ops.concat(z, 1), x)

        return out

    def fuse(self):
        print("IDetect.fuse")
        # fuse ImplicitA and Convolution
        for i in range(len(self.m)):
            c1, c2, _, _ = self.m[i].weight.shape
            c1_, c2_, _, _ = self.ia[i].implicit.shape
            _value = self.m[i].bias + ops.matmul(self.m[i].weight.reshape(c1, c2),
                                                 self.ia[i].implicit.reshape(c2_, c1_)).squeeze(1)
            self.m[i].bias = ops.assign(self.m[i].bias, _value)

        # fuse ImplicitM and Convolution
        for i in range(len(self.m)):
            c1, c2, _, _ = self.im[i].implicit.shape
            self.m[i].bias = ops.assign(self.m[i].bias, self.m[i].bias * self.im[i].implicit.reshape(c2))
            self.m[i].weight = ops.assign(self.m[i].weight, self.m[i].weight * self.im[i].implicit.transpose(0, 1))
            # self.m[i].bias *= self.im[i].implicit.reshape(c2)
            # self.m[i].weight *= self.im[i].implicit.transpose(0, 1)

    @staticmethod
    def _make_grid(nx=20, ny=20, dtype=ms.float32):
        yv, xv = ops.meshgrid([mnp.arange(ny), mnp.arange(nx)])
        return ops.cast(ops.stack((xv, yv), 2).view((1, 1, ny, nx, 2)), dtype)

    def convert(self, z):
        z = ops.concat(z, 1)
        box = z[:, :, :4]
        conf = z[:, :, 4:5]
        score = z[:, :, 5:]
        score *= conf
        convert_matrix = get_convert_matrix()
        box = ops.matmul(box, convert_matrix)
        return (box, score)

class LayerParam:
    def __init__(self, i, f, type, np):
        self.i = i
        self.f = f
        self.type = type
        self.np = np

def parse_model(d, ch, sync_bn=False):  # model_dict, input_channels(3)
    _SYNC_BN = sync_bn
    print('\n%3s%18s%3s%10s  %-40s%-30s' % ('', 'from', 'n', 'params', 'module', 'arguments'))
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    layers_param = []
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except:
                pass

        n = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in [nn.Conv2d, Conv, RepConv, SPPCSPC]:
            c1, c2 = ch[f], args[0]
            if c2 != no:  # if not output
                c2 = make_divisible(c2 * gw, 8)

            args = [c1, c2, *args[1:]]
            if m in [SPPCSPC,]:
                args.insert(2, n)  # number of repeats
                n = 1
        elif m in (nn.BatchNorm2d, nn.SyncBatchNorm):
            args = [ch[f]]
        elif m is Concat:
            c2 = sum([ch[x] for x in f])
        elif m in [IDetect,]:
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
        else:
            c2 = ch[f]

        m_ = nn.SequentialCell([m(*args) for _ in range(n)]) if n > 1 else m(*args)

        t = str(m) # module type
        np = sum([x.size for x in m_.get_parameters()])  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        layers_param.append((i, f, t, np))
        print('%3s%18s%3s%10.0f  %-40s%-30s' % (i, f, n, np, t, args))  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.CellList(layers), sorted(save), layers_param


class ModelEMA:
    """ Model Exponential Moving Average from https://github.com/rwightman/pytorch-image-models
    Keep a moving average of everything in the model state_dict (parameters and buffers).
    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    A smoothed version of the weights is necessary for some training schemes to perform well.
    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    """

    def __init__(self, model, decay=0.9999, updates=0):
        # Create EMA
        model.set_train(False)
        self.ema_model = model
        self.updates = updates  # number of EMA updates
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000))  # decay exponential ramp (to help early epochs)

    def update(self, model):
        # Update EMA parameters
        self.updates += 1
        d = self.decay(self.updates)

        train_parameters = model.parameters_dict()
        for k, v in self.ema_model.parameters_dict():
            v_np = v.asnumpy()
            v_np *= d
            v_np += (1. - d) * train_parameters[k].asnumpy()
            v.set_data(Tensor(v_np, v.dtype))

    def copy_attr(self, a, b, include=(), exclude=()):
        # Copy attributes from b to a, options to only include [...] and to exclude [...]
        for k, v in b.__dict__.items():
            if (len(include) and k not in include) or k.startswith('_') or k in exclude:
                continue
            else:
                setattr(a, k, v)

    def update_attr(self, model, include=(), exclude=('process_group', 'reducer')):
        # Update EMA attributes
        self.copy_attr(self.ema_model, model, include, exclude)