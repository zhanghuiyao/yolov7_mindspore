import math
import numpy as np
from copy import deepcopy

import mindspore as ms
import mindspore.numpy as mnp
from mindspore import nn, ops, Tensor
from mindspore.common.initializer import HeUniform

from .layers import *

__all__ = ['Model']

# import os, sys
# dir_path = os.path.dirname(os.path.realpath(__file__))
# parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
# sys.path.insert(0, parent_dir_path)


@ops.constexpr
def _get_h_w_list(ratio, gs, hw):
    return tuple([math.ceil(x * ratio / gs) * gs for x in hw])

class IDetect(nn.Cell):
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
        self.anchors = ms.Parameter(Tensor(anchors, ms.float32).view(self.nl, -1, 2),
                                    requires_grad=False) # shape(nl,na,2)
        self.anchor_grid = ms.Parameter(Tensor(anchors, ms.float32).view(self.nl, 1, -1, 1, 1, 2),
                                        requires_grad=False) # shape(nl,1,na,1,1,2)
        self.convert_matrix = Tensor(np.array([[1, 0, 1, 0],
                                               [0, 1, 0, 1],
                                               [-0.5, 0, 0.5, 0],
                                               [0, -0.5, 0, 0.5]]), dtype=ms.float32)

        self.m = nn.CellList([nn.Conv2d(x, self.no * self.na, 1,
                                        pad_mode="valid",
                                        has_bias=True,
                                        weight_init=HeUniform(negative_slope=math.sqrt(5)),
                                        bias_init=init_bias((self.no * self.na, x, 1, 1))) for x in ch])  # output conv


        self.ia = nn.CellList([ImplicitA(x) for x in ch])
        self.im = nn.CellList([ImplicitM(self.no * self.na) for _ in ch])

    def construct(self, x):
        z = ()  # inference output
        outs = ()
        for i in range(self.nl):
            out = self.m[i](self.ia[i](x[i]))  # conv
            out = self.im[i](out)
            bs, _, ny, nx = out.shape # (bs,255,20,20)
            out = ops.Transpose()(out.view(bs, self.na, self.no, ny, nx), (0, 1, 3, 4, 2)) # (bs,3,20,20,85)
            outs += (out,)

            if not self.training:  # inference
                # xv, yv = ops.meshgrid((mnp.arange(ny), mnp.arange(nx)))
                # grid_tensor = ops.cast(ops.stack((xv, yv), 2).view((1, 1, ny, nx, 2)), out.dtype)
                grid_tensor = self._make_grid(nx, ny, out.dtype)

                y = ops.Sigmoid()(out)
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + grid_tensor) * self.stride[i]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                z += (y.view(bs, -1, self.no),)

        return outs if self.training else (ops.concat(z, 1), outs)

    def fuseforward(self, x):
        z = ()  # inference output
        self.training |= self.export
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = ops.transpose(x[i].view(bs, self.na, self.no, ny, nx), (0, 1, 3, 4, 2))
            x[i] = x[i]

            if not self.training:  # inference
                grid_tensor = self._make_grid(nx, ny, x.dtype)
                y = ops.Sigmoid()(x[i])
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + grid_tensor) * self.stride[i]  # xy
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
        xv, yv = ops.meshgrid((mnp.arange(ny), mnp.arange(nx)))
        return ops.cast(ops.stack((xv, yv), 2).view((1, 1, ny, nx, 2)), dtype)

    def convert(self, z):
        z = ops.concat(z, 1)
        box = z[:, :, :4]
        conf = z[:, :, 4:5]
        score = z[:, :, 5:]
        score *= conf
        box = ops.matmul(box, self.convert_matrix)
        return (box, score)

class IAuxDetect(nn.Cell):
    def __init__(self, nc=80, anchors=(), ch=()):  # detection layer
        super(IAuxDetect, self).__init__()
        self.stride = None  # strides computed during build
        self.export = False  # onnx export
        self.end2end = False
        self.include_nms = False
        self.concat = False

        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors

        self.anchors = ms.Parameter(Tensor(anchors, ms.float32).view(self.nl, -1, 2),
                                    requires_grad=False)  # shape(nl,na,2)
        self.anchor_grid = ms.Parameter(Tensor(anchors, ms.float32).view(self.nl, 1, -1, 1, 1, 2),
                                        requires_grad=False)  # shape(nl,1,na,1,1,2)
        self.convert_matrix = Tensor([[1, 0, 1, 0],
                                      [0, 1, 0, 1],
                                      [-0.5, 0, 0.5, 0],
                                      [0, -0.5, 0, 0.5]], dtype=ms.float32)

        self.m = nn.CellList([nn.Conv2d(x, self.no * self.na, 1,
                                        pad_mode="valid",
                                        has_bias=True,
                                        weight_init=HeUniform(negative_slope=math.sqrt(5)),
                                        bias_init=init_bias((self.no * self.na, x, 1, 1))) for x in ch[:self.nl]])  # output conv
        self.m2 = nn.CellList([nn.Conv2d(x, self.no * self.na, 1,
                                         pad_mode="valid",
                                         has_bias=True,
                                         weight_init=HeUniform(negative_slope=math.sqrt(5)),
                                         bias_init=init_bias((self.no * self.na, x, 1, 1))) for x in ch[self.nl:]])  # output conv

        self.ia = nn.CellList([ImplicitA(x) for x in ch[:self.nl]])
        self.im = nn.CellList([ImplicitM(self.no * self.na) for _ in ch[:self.nl]])

    def construct(self, x):
        z = ()  # inference output
        outs_1 = ()
        outs_2 = ()
        # self.training |= self.export
        for i in range(self.nl):
            out1 = self.m[i](self.ia[i](x[i])) # conv
            out1 = self.im[i](out1)
            bs, _, ny, nx = out1.shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            out1 = ops.Transpose()(out1.view(bs, self.na, self.no, ny, nx), (0, 1, 3, 4, 2))
            outs_1 += (out1,)

            out2 = self.m2[i](x[i + self.nl])
            out2 = ops.Transpose()(out2.view(bs, self.na, self.no, ny, nx), (0, 1, 3, 4, 2))
            outs_2 += (out2,)

            if not self.training:  # inference
                grid_tensor = self._make_grid(nx, ny, out1.dtype)

                y = ops.Sigmoid()(out1)
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + grid_tensor) * self.stride[i]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                z += (y.view(bs, -1, self.no),)
        outs = outs_1 + outs_2
        return outs if self.training else (ops.concat(z, 1), outs_1)

    def fuseforward(self, x):
        z = ()  # inference output
        outs_1 = ()
        # self.training |= self.export
        for i in range(self.nl):
            out1 = self.m[i](x[i])  # conv
            bs, _, ny, nx = out1.shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            out1 = ops.Transpose()(out1.view(bs, self.na, self.no, ny, nx), (0, 1, 3, 4, 2))
            outs_1 += (out1,)

            if not self.training:  # inference
                grid_tensor = self._make_grid(nx, ny, out1.dtype)

                y = ops.Sigmoid()(out1)
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + grid_tensor) * self.stride[i]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                z += (y.view(bs, -1, self.no),)

        if self.training:
            outs = outs_1
        elif self.end2end:
            outs = ops.concat(z, 1)
        elif self.include_nms:
            z = self.convert(z)
            outs = (z,)
        elif self.concat:
            outs = ops.concat(z, 1)
        else:
            outs = (ops.concat(z, 1), outs_1)

        return outs

    def fuse(self):
        print("IAuxDetect.fuse")
        # fuse ImplicitA and Convolution
        for i in range(len(self.m)):
            c1, c2, _, _ = self.m[i].weight.shape
            c1_, c2_, _, _ = self.ia[i].implicit.shape
            self.m[i].bias += ops.matmul(self.m[i].weight.reshape(c1, c2),
                                         self.ia[i].implicit.reshape(c2_, c1_)).squeeze(1)

        # fuse ImplicitM and Convolution
        for i in range(len(self.m)):
            c1, c2, _, _ = self.im[i].implicit.shape
            self.m[i].bias *= self.im[i].implicit.reshape(c2)
            self.m[i].weight *= self.im[i].implicit.swapaxes(0, 1)

    @staticmethod
    def _make_grid(nx=20, ny=20, dtype=ms.float32):
        xv, yv = ops.meshgrid((mnp.arange(ny), mnp.arange(nx)))
        return ops.cast(ops.stack((xv, yv), 2).view((1, 1, ny, nx, 2)), dtype)

    def convert(self, z):
        z = ops.concat(z, 1)
        box = z[:, :, :4]
        conf = z[:, :, 4:5]
        score = z[:, :, 5:]
        score *= conf
        box = ops.matmul(box, self.convert_matrix)
        return (box, score)

class Model(nn.Cell):
    def __init__(self, opt, ch=3, nc=None, sync_bn=False):  # model, input channels, number of classes
        super(Model, self).__init__()
        self.traced = False
        self.opt = opt

        # Define model
        if nc and nc != opt.nc:
            print(f"Overriding model.yaml nc={opt.nc} with nc={nc}")
            self.nc = nc  # override yaml value
        self.model, self.save, self.layers_param = parse_model(deepcopy(self.opt), ch=[ch], sync_bn=sync_bn)
        self.names = [str(i) for i in range(opt.nc)]  # default names
        # print([x.shape for x in self.forward(torch.zeros(1, ch, 64, 64))])

        # Recompute
        if opt is not None:
            if opt.recompute and opt.recompute_layers > 0:
                for i in range(opt.recompute_layers):
                    self.model[i].recompute()
                print(f"Turn on recompute, and the results of the first {opt.recompute_layers} layers "
                      f"will be recomputed.")

        # Build strides, anchors
        m = self.model[-1]  # Detect()
        if isinstance(m, IDetect):
            m.stride = Tensor(np.array(self.opt.stride), ms.int32)
            check_anchor_order(m)
            m.anchors /= m.stride.view(-1, 1, 1)
            self.stride = m.stride
            self.stride_max = int(np.array(self.opt.stride).max())
            self._initialize_biases()  # only run once
            # print('Strides: %s' % m.stride.tolist())
        if isinstance(m, IAuxDetect):
            m.stride = Tensor(np.array(self.opt.stride), ms.int32)
            check_anchor_order(m)
            m.anchors /= m.stride.view(-1, 1, 1)
            self.stride = m.stride
            self.stride_max = int(np.array(self.opt.stride).max())
            self._initialize_aux_biases()  # only run once
            # print('Strides: %s' % m.stride.tolist())

        # Init weights, biases
        initialize_weights(self.model)

    def scale_img(self, img, ratio=1.0, same_shape=False, gs=32):  # img(16,3,256,416)
        # scales img(bs,3,y,x) by ratio constrained to gs-multiple
        if ratio == 1.0:
            return img
        else:
            h, w = img.shape[2:]
            s = (int(h * ratio), int(w * ratio))  # new size
            img = ops.ResizeBilinear(size=s, align_corners=False)(img)
            if not same_shape:  # pad/crop img
                h, w = _get_h_w_list(ratio, gs, (h, w))

            # img = F.pad(img, [0, w - s[1], 0, h - s[0]], value=0.447)  # value = imagenet mean
            img = ops.pad(img, ((0, 0), (0, 0), (0, w - s[1]), (0, h - s[0])))
            img[:, :, -(w - s[1]):, :] = 0.447
            img[:, :, :, -(h - s[0]):] = 0.447
            return img

    def construct(self, x, augment=False):
        if augment:
            img_size = x.shape[-2:]  # height, width
            s = (1, 0.83, 0.67)  # scales
            f = (None, 3, None)  # flips (2-ud, 3-lr)
            y = ()  # outputs
            for si, fi in zip(s, f):
                xi = self.scale_img(ops.ReverseV2(fi)(x) if fi else x, si, gs=self.stride_max)
                # xi = self.scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
                yi = self.forward_once(xi)[0]  # forward
                # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
                yi[..., :4] /= si  # de-scale
                if fi == 2:
                    yi[..., 1] = img_size[0] - yi[..., 1]  # de-flip ud
                elif fi == 3:
                    yi[..., 0] = img_size[1] - yi[..., 0]  # de-flip lr
                y += (yi,)
            return ops.concat(y, 1) # augmented inference, train
        else:
            return self.forward_once(x)  # single-scale inference, train

    def forward_once(self, x):
        y, dt = (), ()  # outputs
        for i in range(len(self.model)):
            m = self.model[i]
            iol, f, _, _ = self.layers_param[i] # iol: index of layers

            if not(isinstance(f, int) and f == -1): # if not from previous layer
                # x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
                if isinstance(f, int):
                    x = y[f]
                else:
                    _x = ()
                    for j in f:
                        if j == -1:
                            _x += (x,)
                        else:
                            _x += (y[j],)
                    x = _x

            if self.traced:
                if isinstance(m, IDetect):
                    break

            # print("index m: ", iol) # print if debug on pynative mode, not available on graph mode.
            x = m(x)  # run

            y += (x if iol in self.save else None,)  # save output

        return x

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            s = s.asnumpy()
            b = mi.bias.view(m.na, -1).asnumpy()  # conv.bias(255) to (3,85)
            b[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else np.log(cf / cf.sum())  # cls
            mi.bias = ops.assign(mi.bias, Tensor(b, ms.float32).view(-1))

    def _initialize_aux_biases(self, cf=None): # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        m = self.model[-1]  # Detect() module
        for mi, mi2, s in zip(m.m, m.m2, m.stride):  # from
            s = s.asnumpy()

            b = mi.bias.view(m.na, -1).asnumpy()  # conv.bias(255) to (3,85)
            b[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else np.log(cf / cf.sum())  # cls
            mi.bias = ops.assign(mi.bias, Tensor(b, ms.float32).view(-1))

            b2 = mi2.bias.view(m.na, -1).asnumpy()  # conv.bias(255) to (3,85)
            b2[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b2[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else np.log(cf / cf.sum())  # cls
            mi2.bias = ops.assign(mi2.bias, Tensor(b2, ms.float32).view(-1))


def parse_model(d, ch, sync_bn=False):  # model_dict, input_channels(3)
    _SYNC_BN = sync_bn
    if _SYNC_BN:
        print('Parse model with Sync BN.')
    print('\n%3s%18s%3s%10s  %-40s%-30s' % ('', 'from', 'n', 'params', 'module', 'arguments'))
    anchors, nc, gd, gw = d.anchors, d.nc, d.depth_multiple, d.width_multiple
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    layers_param = []
    for i, (f, n, m, args) in enumerate(d.backbone + d.head):  # from, number, module, args
        kwargs = {}
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except:
                pass

        n = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in (nn.Conv2d, Conv, RepConv, DownC, SPPCSPC):
            c1, c2 = ch[f], args[0]
            if c2 != no:  # if not output
                c2 = math.ceil(c2 * gw / 8) * 8

            args = [c1, c2, *args[1:]]
            if m in (Conv, RepConv):
                kwargs["sync_bn"] = sync_bn
            if m in (DownC, SPPCSPC,):
                args.insert(2, n)  # number of repeats
                n = 1
        elif m in (nn.BatchNorm2d, nn.SyncBatchNorm):
            args = [ch[f]]
        elif m in (Concat,):
            c2 = sum([ch[x] for x in f])
        elif m is Shortcut:
            c2 = ch[f[0]]
        elif m in (IDetect, IAuxDetect):
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
        elif m is ReOrg:
            c2 = ch[f] * 4
        else:
            c2 = ch[f]

        m_ = nn.SequentialCell([m(*args, **kwargs) for _ in range(n)]) if n > 1 else m(*args, **kwargs)

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

if __name__ == '__main__':
    from mindspore import context
    from utils.config import parse_args
    context.set_context(mode=context.GRAPH_MODE, pynative_synchronize=True)
    opt = parse_args("train")
    model = Model(opt, ch=3, nc=80)
    for p in model.trainable_params():
        print(p.name)
    # model.set_train(True)
    # x = Tensor(np.random.randn(1, 3, 160, 160), ms.float32)
    # pred = model(x)
    # print(pred)