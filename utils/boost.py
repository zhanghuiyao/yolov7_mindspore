from mindspore import ops
from mindspore import nn
from mindspore import boost
from mindspore import context
from mindspore.ops import functional as F
from mindspore.common import dtype as mstype
from mindspore.nn.wrap.cell_wrapper import TrainOneStepCell
from mindspore.boost.boost_cell_wrapper import BoostTrainOneStepCell
from mindspore.nn.wrap.cell_wrapper import _TrainPipelineAccuStepCell, _pipeline_clear_grad
from mindspore.nn.wrap.loss_scale import _TrainPipelineWithLossScaleCell
from mindspore.train.amp import validator, _check_level, _check_kwargs, _config_level, \
    _do_keep_batchnorm_fp32, auto_mixed_precision, _add_loss_network, _get_pipeline_stages


class _BoostTrainPipelineAccuStepCell(_TrainPipelineAccuStepCell):
    def __init__(self, network, ema, optimizer, amp_loss_scaler, sens=1.0):
        super(_BoostTrainPipelineAccuStepCell, self).__init__(network, optimizer, sens=sens)
        self.ema = ema
        self.use_loss_scaler = False if amp_loss_scaler is None else True
        self.amp_loss_scaler = amp_loss_scaler
        print(f"[INFO] Enable loss scale: {self.use_loss_scaler}", flush=True)

    def construct(self, *inputs):
        weights = self.weights
        loss = self.network(*inputs)
        sens = ops.Fill()(ops.DType()(loss), ops.Shape()(loss), self.sens)
        grads = self.grad(self.network, weights)(*inputs, sens)
        if self.use_loss_scaler:
            grads = self.amp_loss_scaler.unscale(grads)
        accu_grads = ops.depend(self.accu_grads, grads)
        if self.opt_shard:
            succ = self.optimizer(grads)
        else:
            succ = self.optimizer(accu_grads)
        loss = ops.depend(loss, succ)
        clear = self.hyper_map(_pipeline_clear_grad, accu_grads, grads)
        loss = ops.depend(loss, clear)
        loss = F.depend(loss, self.ema.update())
        return loss


class _TrainOneStepCell(TrainOneStepCell):
    def __init__(self, network, ema, optimizer, amp_loss_scaler, sens=1.0):
        super(_TrainOneStepCell, self).__init__(network, optimizer, sens=sens)
        self.ema = ema
        self.use_loss_scaler = False if amp_loss_scaler is None else True
        self.amp_loss_scaler = amp_loss_scaler
        print(f"[INFO] Enable loss scale: {self.use_loss_scaler}", flush=True)

    def construct(self, *inputs):
        loss = self.network(*inputs)
        sens = F.fill(loss.dtype, loss.shape, self.sens)
        grads = self.grad(self.network, self.weights)(*inputs, sens)
        grads = self.grad_reducer(grads)
        if self.use_loss_scaler:
            grads = self.amp_loss_scaler.unscale(grads)
        loss = F.depend(loss, self.optimizer(grads))
        loss = F.depend(loss, self.ema.update())
        return loss


class _BoostTrainOneStepCell(BoostTrainOneStepCell):
    def __init__(self, network, ema, optimizer, amp_loss_scaler, sens=1.0):
        super(_BoostTrainOneStepCell, self).__init__(network, optimizer, sens)
        self.ema = ema
        self.use_loss_scaler = False if amp_loss_scaler is None else True
        self.amp_loss_scaler = amp_loss_scaler
        print(f"[INFO] Enable loss scale: {self.use_loss_scaler}", flush=True)

    def construct(self, *inputs):
        if self.freeze:
            loss = self.gradient_freeze_process(*inputs)
        else:
            loss = self.network(*inputs)
            sens = F.fill(loss.dtype, loss.shape, self.sens)
            grads = self.grad(self.network, self.weights)(*inputs, sens)
            grads = self.grad_reducer(grads)
            if self.use_loss_scaler:
                grads = self.amp_loss_scaler.unscale(grads)
            if self.use_grad_accumulation:
                loss = self.gradient_accumulation_process(loss, grads, sens, *inputs)
            else:
                if self.enable_dim_reduce:
                    loss = F.depend(loss, self.dim_reduce(loss, grads, sens, self.weights, self.weights_clone, *inputs))
                elif self.enable_adasum:
                    loss = F.depend(loss, self.adasum_process(loss, grads))
                else:
                    loss = F.depend(loss, self.optimizer(grads))

        loss = F.depend(loss, self.ema.update())
        return loss


def build_train_network(network, ema, optimizer, loss_fn=None, level='O0', boost_level='O0',
                        amp_loss_scaler=None, sens=1.0, **kwargs):
    validator.check_value_type('optimizer', optimizer, (nn.Optimizer, boost.FreezeOpt,
                                                        nn.AdaSumByGradWrapCell, nn.AdaSumByDeltaWeightWrapCell))

    level, enable_boost = _check_level(level, boost_level)

    _check_kwargs(kwargs)
    config = dict(_config_level.get(level), **kwargs)

    if config["cast_model_type"] == mstype.float16:
        network.to_float(mstype.float16)

        if config["keep_batchnorm_fp32"]:
            _do_keep_batchnorm_fp32(network)
    elif not config["keep_batchnorm_fp32"] and level == "O2":
        network.to_float(mstype.float16)
    elif config["cast_model_type"] == mstype.float32 and level in ("O2", "O3"):
        pass
    else:
        auto_mixed_precision(network, level)

    if loss_fn:
        network = _add_loss_network(network, loss_fn, config["cast_model_type"])

    loss_scale = 1.0
    if config["loss_scale_manager"] is not None:
        loss_scale_manager = config["loss_scale_manager"]
        loss_scale = loss_scale_manager.get_loss_scale()
        update_cell = loss_scale_manager.get_update_cell()
        if update_cell is not None:
            # only cpu not support `TrainOneStepWithLossScaleCell` for control flow.
            if not context.get_context("enable_ge") and context.get_context("device_target") == "CPU":
                raise ValueError("Only `loss_scale_manager=None` or "
                                 "`loss_scale_manager=FixedLossScaleManager(drop_overflow_update=False)`"
                                 "are supported on device `CPU`. ")
            if _get_pipeline_stages() > 1:
                network = _TrainPipelineWithLossScaleCell(network, optimizer,
                                                          scale_sense=update_cell).set_train()
            elif enable_boost:
                network = boost.BoostTrainOneStepWithLossScaleCell(network, optimizer,
                                                                   scale_sense=update_cell).set_train()
            else:
                network = nn.TrainOneStepWithLossScaleCell(network, optimizer,
                                                           scale_sense=update_cell).set_train()
            return network
    sens = sens if config["loss_scale_manager"] is not None else loss_scale
    if _get_pipeline_stages() > 1:
        network = _BoostTrainPipelineAccuStepCell(network, ema, optimizer, amp_loss_scaler, sens=sens).set_train()
    elif enable_boost:
        network = _BoostTrainOneStepCell(network, ema, optimizer, amp_loss_scaler, sens=sens).set_train()
    else:
        network = _TrainOneStepCell(network, ema, optimizer, amp_loss_scaler, sens=sens).set_train()
    return network
