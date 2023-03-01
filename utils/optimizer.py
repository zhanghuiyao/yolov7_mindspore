import math
import numpy as np

import mindspore as ms
from mindspore import nn
import mindspore.common.dtype as mstype
from mindspore.common.tensor import Tensor
from mindspore._checkparam import Validator
from mindspore.common.api import ms_function
from mindspore.common.parameter import Parameter
from mindspore.common.parameter import ParameterTuple
from mindspore.ops import functional as F, composite as C, operations as P
from mindspore.nn.optim.optimizer import opt_init_args_register


def one_cycle(y1=0.0, y2=1.0, steps=100):
    # lambda function for sinusoidal ramp from y1 to y2
    return lambda x: ((1 - math.cos(x * math.pi / steps)) / 2) * (y2 - y1) + y1


def get_group_param(model):
    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in model.cells_and_names():
        if hasattr(v, 'bias') and isinstance(v.bias, ms.Parameter):
            pg2.append(v.bias)  # biases
        elif hasattr(v, 'beta') and isinstance(v.beta, ms.Parameter):
            pg2.append(v.beta)  # bn biases
        if isinstance(v, nn.BatchNorm2d):
            pg0.append(v.gamma)  # no decay
        elif hasattr(v, 'weight') and isinstance(v.weight, ms.Parameter):
            pg1.append(v.weight)  # apply decay
        if hasattr(v, 'im'):
            if hasattr(v.im, 'implicit'):
                pg0.append(v.im.implicit)
            else:
                for iv in v.im:
                    pg0.append(iv.implicit)
        if hasattr(v, 'imc'):
            if hasattr(v.imc, 'implicit'):
                pg0.append(v.imc.implicit)
            else:
                for iv in v.imc:
                    pg0.append(iv.implicit)
        if hasattr(v, 'imb'):
            if hasattr(v.imb, 'implicit'):
                pg0.append(v.imb.implicit)
            else:
                for iv in v.imb:
                    pg0.append(iv.implicit)
        if hasattr(v, 'imo'):
            if hasattr(v.imo, 'implicit'):
                pg0.append(v.imo.implicit)
            else:
                for iv in v.imo:
                    pg0.append(iv.implicit)
        if hasattr(v, 'ia'):
            if hasattr(v.ia, 'implicit'):
                pg0.append(v.ia.implicit)
            else:
                for iv in v.ia:
                    pg0.append(iv.implicit)
        if hasattr(v, 'attn'):
            if hasattr(v.attn, 'logit_scale'):
                pg0.append(v.attn.logit_scale)
            if hasattr(v.attn, 'q_bias'):
                pg0.append(v.attn.q_bias)
            if hasattr(v.attn, 'v_bias'):
                pg0.append(v.attn.v_bias)
            if hasattr(v.attn, 'relative_position_bias_table'):
                pg0.append(v.attn.relative_position_bias_table)
        if hasattr(v, 'rbr_dense'):
            if hasattr(v.rbr_dense, 'weight_rbr_origin'):
                pg0.append(v.rbr_dense.weight_rbr_origin)
            if hasattr(v.rbr_dense, 'weight_rbr_avg_conv'):
                pg0.append(v.rbr_dense.weight_rbr_avg_conv)
            if hasattr(v.rbr_dense, 'weight_rbr_pfir_conv'):
                pg0.append(v.rbr_dense.weight_rbr_pfir_conv)
            if hasattr(v.rbr_dense, 'weight_rbr_1x1_kxk_idconv1'):
                pg0.append(v.rbr_dense.weight_rbr_1x1_kxk_idconv1)
            if hasattr(v.rbr_dense, 'weight_rbr_1x1_kxk_conv2'):
                pg0.append(v.rbr_dense.weight_rbr_1x1_kxk_conv2)
            if hasattr(v.rbr_dense, 'weight_rbr_gconv_dw'):
                pg0.append(v.rbr_dense.weight_rbr_gconv_dw)
            if hasattr(v.rbr_dense, 'weight_rbr_gconv_pw'):
                pg0.append(v.rbr_dense.weight_rbr_gconv_pw)
            if hasattr(v.rbr_dense, 'vector'):
                pg0.append(v.rbr_dense.vector)
    return pg0, pg1, pg2


def get_lr(opt, hyp, per_epoch_size):
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    # https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#OneCycleLR
    init_lr, warmup_bias_lr, warmup_epoch, lrf = \
        hyp['lr0'], hyp['warmup_bias_lr'], hyp['warmup_epochs'], hyp['lrf']
    total_epoch, linear_lr = opt.epochs, opt.linear_lr
    if opt.optimizer == "sgd":
        with_momentum = True
    elif opt.optimizer == "momentum":
        with_momentum = True
    elif opt.optimizer == "adam":
        with_momentum = False
    elif opt.optimizer == "thor":  # not use this lr
        with_momentum = False
    else:
        raise NotImplementedError

    if linear_lr:
        lf = lambda x: (1 - x / (total_epoch - 1)) * (1.0 - lrf) + lrf  # linear
    else:
        lf = one_cycle(1, lrf, total_epoch)  # cosine 1->hyp['lrf'] #1 -> 0.1

    lr_pg0, lr_pg1, lr_pg2 = [], [], []
    momentum_pg = []
    warmup_steps = max(round(warmup_epoch * per_epoch_size), 1000)
    warmup_bias_steps_first = min(max(round(3 * per_epoch_size), 1000), warmup_steps)
    warmup_bias_lr_first = np.interp(warmup_bias_steps_first, [0, warmup_steps], [0.0, init_lr])
    xi = [0, warmup_steps]
    momentum_after_warmup = np.interp(warmup_steps, xi, [hyp['warmup_momentum'], hyp['momentum']])
    for i in range(total_epoch * per_epoch_size):
        cur_epoch = i // per_epoch_size
        _lr = init_lr * lf(cur_epoch)
        if i < warmup_steps:
            lr_pg0.append(np.interp(i, xi, [0.0, _lr]))
            lr_pg1.append(np.interp(i, xi, [0.0, _lr]))
            lr_pg2.append(np.interp(i,
                                    [0, warmup_bias_steps_first, warmup_steps],
                                    [warmup_bias_lr, warmup_bias_lr_first, _lr]))
            if with_momentum:
                momentum_pg.append(np.interp(i, xi, [hyp['warmup_momentum'], hyp['momentum']]))
        else:
            lr_pg0.append(_lr)
            lr_pg1.append(_lr)
            lr_pg2.append(_lr)
            if with_momentum:
                momentum_pg.append(momentum_after_warmup)

    return lr_pg0, lr_pg1, lr_pg2, momentum_pg, warmup_steps


# Thor
def get_thor_lr(global_step, lr_init, decay, total_epochs, steps_per_epoch, decay_epochs=100):
    """get_model_lr"""
    lr_each_step = []
    total_steps = steps_per_epoch * total_epochs
    for i in range(total_steps):
        epoch = (i + 1) / steps_per_epoch
        base = (1.0 - float(epoch) / total_epochs) ** decay
        lr_local = lr_init * base
        if epoch >= decay_epochs:
            lr_local = lr_local * 0.5
        if epoch >= decay_epochs + 1:
            lr_local = lr_local * 0.5
        lr_each_step.append(lr_local)
    current_step = global_step
    lr_each_step = np.array(lr_each_step).astype(np.float32)
    learning_rate = lr_each_step[current_step:]
    return learning_rate


def get_thor_damping(global_step, damping_init, decay_rate, total_epochs, steps_per_epoch):
    """get_model_damping"""
    damping_each_step = []
    total_steps = steps_per_epoch * total_epochs
    for step in range(total_steps):
        epoch = (step + 1) / steps_per_epoch
        damping_here = damping_init * (decay_rate ** (epoch / 10))
        damping_each_step.append(damping_here)
    current_step = global_step
    damping_each_step = np.array(damping_each_step).astype(np.float32)
    damping_now = damping_each_step[current_step:]
    return damping_now


_momentum_opt = C.MultitypeFuncGraph("momentum_opt")


@_momentum_opt.register("Function", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Bool", "Bool")
def _tensor_run_opt_ext(opt, momentum, learning_rate, gradient, weight, moment, ps_parameter, cache_enable):
    """Apply momentum optimizer to the weight parameter using Tensor."""
    if ps_parameter and not cache_enable:
        op_shape = P.Shape()
        _ps_pull = P.Pull()
        _ps_push = P.Push("ApplyMomentum", [])
        shapes = (op_shape(learning_rate), op_shape(gradient), op_shape(momentum))
        success = F.depend(True, _ps_pull(_ps_push((learning_rate, gradient, momentum), shapes), weight))
    else:
        success = F.depend(True, opt(weight, moment, learning_rate, gradient, momentum))
    return success


@_momentum_opt.register("Function", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Bool", "Bool",
                        "Function", "Bool")
def _tensor_run_opt_ext_dist(opt, momentum, learning_rate, gradient, weight, moment, ps_parameter, cache_enable,
                             distributed_opt, use_flag):
    """Apply momentum optimizer to the weight parameter using Tensor."""
    if use_flag:
        success = F.depend(True, distributed_opt(weight, moment, learning_rate, gradient, momentum))
    elif ps_parameter and not cache_enable:
        op_shape = P.Shape()
        _ps_pull = P.Pull()
        _ps_push = P.Push("ApplyMomentum", [])
        shapes = (op_shape(learning_rate), op_shape(gradient), op_shape(momentum))
        success = F.depend(True, _ps_pull(_ps_push((learning_rate, gradient, momentum), shapes), weight))
    else:
        success = F.depend(True, opt(weight, moment, learning_rate, gradient, momentum))
    return success


class YoloMomentum(nn.Optimizer):
    @opt_init_args_register
    def __init__(self, params, learning_rate, momentum, weight_decay=0.0, loss_scale=1.0, use_nesterov=False):
        super(YoloMomentum, self).__init__(learning_rate, params, weight_decay, loss_scale)
        # Validator.check_value_type("momentum", momentum, [float], self.cls_name)
        if isinstance(momentum, float) and momentum < 0.0:
            raise ValueError("For 'Momentum', the argument 'momentum' must be at least 0.0, "
                             "but got {}".format(momentum))
        if isinstance(momentum, (float, int)):
            self.momentum = Parameter(Tensor(momentum, mstype.float32), name="momentum")
            self.list_moment = False
        elif isinstance(momentum, (list, tuple)):
            self.momentum = Parameter(np.array(momentum).astype(np.float32), name="momentum")
            self.list_moment = True
        self.params = self._parameters
        self.use_nesterov = Validator.check_bool(use_nesterov)
        self.moments = self.params.clone(prefix="moments", init='zeros')
        self.opt = P.ApplyMomentum(use_nesterov=self.use_nesterov)

        self.distributed_opts, self.use_distributed_opt_flags = \
            self._get_distributed_optimizer_list("momentum", use_nesterov=self.use_nesterov)
        self.use_dist_optimizer = self._use_distibuted_optimizer()
        self.gather = P.Gather()

    @ms_function
    def construct(self, gradients):
        params = self.params
        moments = self.moments
        if self.list_moment:
            momentum = self.gather(self.momentum, self.global_step, 0)
        else:
            momentum = self.momentum
        gradients = self.flatten_gradients(gradients)
        gradients = self.decay_weight(gradients)
        gradients = self.gradients_centralization(gradients)
        gradients = self.scale_grad(gradients)
        lr = self.get_lr()
        if self.use_dist_optimizer:
            if self.is_group_lr:
                success = self.hyper_map_reverse(F.partial(_momentum_opt, self.opt, momentum),
                                                 lr, gradients, params, moments, self.ps_parameters, self.cache_enable,
                                                 self.distributed_opts, self.use_distributed_opt_flags)
            else:
                success = self.hyper_map_reverse(F.partial(_momentum_opt, self.opt, momentum, lr),
                                                 gradients, params, moments, self.ps_parameters, self.cache_enable,
                                                 self.distributed_opts, self.use_distributed_opt_flags)
        else:
            if self.is_group_lr:
                success = self.hyper_map_reverse(F.partial(_momentum_opt, self.opt, momentum),
                                                 lr, gradients, params, moments, self.ps_parameters, self.cache_enable)
            else:
                success = self.hyper_map_reverse(F.partial(_momentum_opt, self.opt, momentum, lr),
                                                 gradients, params, moments, self.ps_parameters, self.cache_enable)
        return success
