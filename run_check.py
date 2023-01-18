import yaml
import glob
import logging
import math
import os
import random
import shutil
import time
import cv2
import hashlib
import multiprocessing
import numpy as np
from itertools import repeat
from PIL import Image, ExifTags
from tqdm import tqdm
from multiprocessing.pool import ThreadPool
from pathlib import Path
from threading import Thread

import mindspore as ms
from mindspore.context import ParallelMode
from mindspore import nn, ops, Tensor, context
from mindspore.communication.management import init, get_rank, get_group_size
# from mindspore.amp import DynamicLossScaler, all_finite
from mindspore.ops import functional as F
from mindspore.ops import composite as C
from mindspore.ops import operations as P

from mindyolo.models.yolo import Model
from mindyolo.models import EMA, ImplicitA, ImplicitM
from mindyolo.models.loss import *
from mindyolo.models.loss import BCEWithLogitsLoss
from mindyolo.utils.optimizer import get_group_param_yolov7, get_lr_yolov7
from mindyolo.utils.dataset import create_dataloader
from mindyolo.utils.general import increment_path, colorstr, check_file, check_img_size
from mindyolo.utils.config import parse_args

class SimpleComputeLoss(nn.Cell):
    # Compute losses
    def __init__(self):
        super(SimpleComputeLoss, self).__init__()
        self.BCE = BCEWithLogitsLoss()

    def construct(self, p, targets):  # predictions, targets
        loss = 0.
        for i in range(len(p)):
            loss += ((p[i] - 0.5) ** 2).mean()
            #loss += self.BCE(p[i].mean().view(1), 0.5 * ops.ones((1,), ms.float32))
        return loss, ops.stop_gradient(ops.stack((loss, loss, loss, loss)))

id2name = {}

def forward_hook_fn(cell_id, inputs, output):
    print(f"[cell] id: {cell_id}, name: {id2name[cell_id]}", flush=True)
    if isinstance(inputs[0], (list, tuple)):
        for i, f_in in enumerate(inputs[0]):
            print(f"[forward in] {i}, shape: {f_in.shape}, mean: {f_in.mean()}, sum: {f_in.sum()}, min: {f_in.min()}, "
                  f"max: {f_in.max()}, has_nan: {ops.isnan(f_in).any()}", flush=True)
            np.save(f"log_hook_result/{id2name[cell_id]}-fi_{i}.npy", f_in.asnumpy())
    else:
        print(f"[forward in], shape: {inputs[0].shape}, mean: {inputs[0].mean()}, sum: {inputs[0].sum()}, min: {inputs[0].min()}, "
              f"max: {inputs[0].max()}, has_nan: {ops.isnan(inputs[0]).any()}", flush=True)
        np.save(f"log_hook_result/{id2name[cell_id]}-fi.npy", inputs[0].asnumpy())
    if isinstance(output, (list, tuple)):
        for j, out in enumerate(output):
            print(f"[forward out] {j}, shape: {out.shape}, mean {out.mean()}, sum: {out.sum()}, min: {out.min()}, "
                  f"max: {out.max()}, has_nan: {ops.isnan(out).any()}", flush=True)
            np.save(f"log_hook_result/{id2name[cell_id]}-fo_{j}.npy", out.asnumpy())
    else:
        print(f"[forward out], shape: {output.shape}, mean: {output.mean()}, sum: {output.sum()}, min: {output.min()}, "
              f"max: {output.max()}, has_nan: {ops.isnan(output).any()}", flush=True)
        np.save(f"log_hook_result/{id2name[cell_id]}-fo.npy", output.asnumpy())

def backward_hook_fn(cell_id, grad_inputs, grad_outputs):
    print(f"[cell] id: {cell_id}, name: {id2name[cell_id]}", flush=True)
    for i, g_in in enumerate(grad_inputs):
        if isinstance(g_in, Tensor):
            print(f"[backward in] {i}, shape: {g_in.shape}, mean: {g_in.mean()}, sum: {g_in.sum()}, min: {g_in.min()}, "
                  f"max: {g_in.max()}, has_nan: {ops.isnan(g_in).any()}", flush=True)
            np.save(f"log_hook_result/{id2name[cell_id]}-bi_{i}.npy", g_in.asnumpy())
        else:
            print(f"[backward in] {i}, grad_in: {g_in}", flush=True)
    for j, g_out in enumerate(grad_outputs):
        if isinstance(g_out, Tensor):
            print(f"[backward out] {j}, shape: {g_out.shape}, mean: {g_out.mean()}, sum: {g_out.sum()}, min: {g_out.min()}, "
                  f"max: {g_out.max()}, has_nan: {ops.isnan(g_out).any()}", flush=True)
            np.save(f"log_hook_result/{id2name[cell_id]}-bo_{j}.npy", g_out.asnumpy())
        else:
            print(f"[backward out] {j}, grad_out: {g_out}", flush=True)

def set_seed(seed=2):
    np.random.seed(seed)
    random.seed(seed)
    ms.set_seed(seed)


def train(opt):
    set_seed()

    save_dir, epochs, batch_size, total_batch_size, rank, freeze = \
        opt.save_dir, opt.epochs, opt.batch_size, opt.total_batch_size, opt.rank, opt.freeze

    nc = 1 if opt.single_cls else int(opt.nc)  # number of classes
    names = ['item'] if opt.single_cls and len(opt.names) != 1 else opt.names  # class names
    assert len(names) == nc, '%g names found for nc=%g dataset in %s' % (len(names), nc, opt.data)  # check

    if opt.enable_modelarts:
        opt.train_set = os.path.join(opt.data_dir, opt.train_set)
        opt.val_set = os.path.join(opt.data_dir, opt.val_set)
        opt.test_set = os.path.join(opt.data_dir, opt.test_set)

    # Directories
    wdir = os.path.join(save_dir, "weights")
    os.makedirs(wdir, exist_ok=True)
    # Save run settings
    with open(os.path.join(save_dir, "opt.yaml"), 'w') as f:
        yaml.dump(vars(opt), f, sort_keys=False)

    # Model
    sync_bn = opt.sync_bn and context.get_context("device_target") == "Ascend" and rank_size > 1
    model = Model(opt, ch=3, nc=nc, sync_bn=sync_bn)  # create
    ema = EMA(model) if opt.ema else None
    if opt.run_eval:
        eval_model = Model(opt, ch=3, nc=nc, sync_bn=sync_bn)
        eval_model.set_train(False)

    # Pretrain
    load_pretrain(model, opt, ema)

    # Hook ms
    """
    for name, cell in model.cells_and_names():
        id2name[cell.cls_name+"("+str(id(cell))+")"] = name
        if isinstance(cell, (nn.SequentialCell, )):
            continue
        print(f"register hook to {name}")
        cell.register_forward_hook(forward_hook_fn)
        if isinstance(cell, (nn.Conv2d, nn.BatchNorm2d, nn.SiLU, ImplicitA, ImplicitM)):
            cell.register_backward_hook(backward_hook_fn)
    """

    # Freeze
    if len(freeze) > 0:
        freeze = [f'model.{x}.' for x in
                  (freeze if len(freeze) > 1 else range(freeze[0]))]  # parameter names to freeze (full or partial)
    for n, p in model.parameters_and_names():
        if any(x in n for x in freeze):
            print('freezing %s' % n)
            p.requires_grad = False

    # Image sizes
    gs = max(int(model.stride.asnumpy().max()), 32)  # grid size (max stride)
    nl = model.model[-1].nl  # number of detection layers (used for scaling opt.obj)
    imgsz, imgsz_test = [check_img_size(x, gs) for x in opt.img_size]  # verify imgsz are gs-multiples
    train_path = opt.train_set
    eval_path = opt.val_set
    dataloader, dataset, per_epoch_size = create_dataloader(train_path, imgsz, batch_size, gs, opt,
                                                            epoch_size=1 if opt.optimizer == "thor" else opt.epochs,
                                                            augment=True,
                                                            rect=opt.rect, rank_size=opt.rank_size, rank=opt.rank,
                                                            num_parallel_workers=opt.num_parallel_workers,
                                                            prefix=colorstr('train: '))
    if opt.run_eval:
        eval_dataloader, eval_dataset, eval_per_epoch_size = create_dataloader(eval_path, imgsz_test, batch_size * 2,
                                                                               gs, opt, epoch_size=1, pad=0.5,
                                                                               rect=False, num_parallel_workers=8,
                                                                               shuffle=False, drop_remainder=False,
                                                                               prefix=colorstr('eval: '))
    mlc = np.concatenate(dataset.labels, 0)[:, 0].max()  # max label class
    assert mlc < nc, 'Label class %g exceeds nc=%g in %s. Possible class labels are 0-%g' % (mlc, nc, opt.data, nc - 1)

    # Optimizer
    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / total_batch_size), 1)  # accumulate loss before optimizing
    opt.weight_decay *= total_batch_size * accumulate / nbs  # scale weight_decay
    print(f"Scaled weight_decay = {opt.weight_decay}")
    if "yolov7" in opt.cfg:
        if opt.optimizer != "thor":
            pg0, pg1, pg2 = get_group_param_yolov7(model)
            lr_pg0, lr_pg1, lr_pg2, momentum_pg, warmup_steps = get_lr_yolov7(opt, per_epoch_size)
            if opt.optimizer in ("sgd", "momentum"):
                assert len(momentum_pg) == warmup_steps
            group_params = [{'params': pg0, 'lr': lr_pg0},
                            {'params': pg1, 'lr': lr_pg1, 'weight_decay': opt.weight_decay},
                            {'params': pg2, 'lr': lr_pg2}]
    else:
        raise NotImplementedError

    print(f"optimizer loss scale is {opt.ms_optim_loss_scale}")
    if opt.optimizer == "sgd":
        optimizer = nn.SGD(group_params, learning_rate=opt.lr0, momentum=opt.momentum, nesterov=True,
                           loss_scale=opt.ms_optim_loss_scale)
    elif opt.optimizer == "momentum":
        optimizer = nn.Momentum(group_params, learning_rate=opt.lr0, momentum=opt.momentum, use_nesterov=True,
                                loss_scale=opt.ms_optim_loss_scale)
    elif opt.optimizer == "adam":
        optimizer = nn.Adam(group_params, learning_rate=opt.lr0, beta1=opt.momentum, beta2=0.999,
                            loss_scale=opt.ms_optim_loss_scale)
    elif opt.optimizer == "thor":
        pass
    else:
        raise NotImplementedError

    # Model parameters
    opt.box *= 3. / nl  # scale to layers
    opt.cls *= nc / 80. * 3. / nl  # scale to classes and layers
    opt.obj *= (imgsz / 640) ** 2 * 3. / nl  # scale to image size and layers
    opt.label_smoothing = opt.label_smoothing

    # amp
    ms.amp.auto_mixed_precision(model, amp_level=opt.ms_amp_level)
    if opt.is_distributed:
        mean = context.get_auto_parallel_context("gradients_mean")
        degree = context.get_auto_parallel_context("device_num")
        grad_reducer = nn.DistributedGradReducer(optimizer.parameters, mean, degree)
    else:
        grad_reducer = ops.functional.identity

    if opt.ms_loss_scaler == "dynamic":
        from mindspore.amp import DynamicLossScaler
        loss_scaler = DynamicLossScaler(2 ** 12, 2, 1000)
    elif opt.ms_loss_scaler == "static":
        from mindspore.amp import StaticLossScaler
        loss_scaler = StaticLossScaler(opt.ms_loss_scaler_value)
    elif opt.ms_loss_scaler == "none":
        loss_scaler = None
    else:
        raise NotImplementedError

    if opt.ms_strategy == "StaticShape":
        train_step = create_train_static_shape_fn_gradoperation(model, optimizer, loss_scaler, grad_reducer,
                                                                amp_level=opt.ms_amp_level,
                                                                overflow_still_update=opt.overflow_still_update,
                                                                sens=opt.ms_grad_sens)
    else:
        raise NotImplementedError


    data_loader = dataloader.create_dict_iterator(output_numpy=True, num_epochs=1)
    s_time_step = time.time()
    s_time_epoch = time.time()
    best_fitness = 0.0
    accumulate_grads = None
    accumulate_cur_step = 0

    model.set_train(True)
    optimizer.set_train(True)

    for i, data in enumerate(data_loader):
        if i < warmup_steps:
            xi = [0, warmup_steps]  # x interp
            accumulate = max(1, np.interp(i, xi, [1, nbs / total_batch_size]).round())
            if opt.optimizer in ("sgd", "momentum"):
                optimizer.momentum = Tensor(momentum_pg[i], ms.float32)
                # print("optimizer.momentum: ", optimizer.momentum.asnumpy())

        cur_epoch = (i // per_epoch_size) + 1
        cur_step = (i % per_epoch_size) + 1
        # imgs, labels, paths = data["img"], data["label_out"], data["img_files"]
        # input_dtype = ms.float32 if opt.ms_amp_level == "O0" else ms.float16
        # imgs, labels = Tensor(imgs, input_dtype), Tensor(labels, input_dtype)

        imgs, labels = np.load("imgs.npy")[0:1, :, :320, :320], np.load("labels.npy")[0:1, ...]
        input_dtype = ms.float32 if opt.ms_amp_level == "O0" else ms.float16
        imgs, labels = Tensor(imgs, input_dtype), Tensor(labels, input_dtype)

        # Multi-scale
        ns = None
        if opt.multi_scale:
            sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs  # size
            sf = sz / max(imgs.shape[2:])  # scale factor
            if sf != 1:
                ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                # imgs = ops.interpolate(imgs, sizes=ns, coordinate_transformation_mode="asymmetric", mode="bilinear")

        # Accumulate Grad
        s_train_time = time.time()
        loss, loss_item, _, grads_finite = train_step(imgs, labels, ns, True)
        # 2. check torch grad
        """
        loss, loss_item, _grad, _ = train_step(imgs, labels, ns, False)
        grad = []
        for i in range(288):
            grad.append(Tensor(np.load(f"../yolov7_official/grad_bak/{i}.npy"), ms.float32))
        new_grad = [g for g in grad]
        new_grad[92] = grad[95]
        new_grad[93] = grad[96]
        new_grad[94] = grad[97]
        new_grad[95] = grad[92]
        new_grad[96] = grad[93]
        new_grad[97] = grad[94]
        optimizer(tuple(new_grad))
        """
        if ema:
            ema.update()
        if loss_scaler:
            loss_scaler.adjust(grads_finite)
            if not grads_finite:
                print("overflow, loss scale adjust to ", loss_scaler.scale_value.asnumpy())

        _p_train_size = ns if ns else imgs.shape[2:]
        print(f"Epoch {epochs}/{cur_epoch}, Step {per_epoch_size}/{cur_step}, size {_p_train_size}, "
              f"fp/bp time cost: {(time.time() - s_train_time) * 1000:.2f} ms")
        if opt.ms_strategy == "StaticCell":
            print(f"Epoch {epochs}/{cur_epoch}, Step {per_epoch_size}/{cur_step}, size {_p_train_size}, "
                  f"loss: {loss.asnumpy():.4f}, step time: {(time.time() - s_time_step) * 1000:.2f} ms")
        else:
            print(f"Epoch {epochs}/{cur_epoch}, Step {per_epoch_size}/{cur_step}, size {_p_train_size}, "
                  f"loss: {loss.asnumpy():.4f}, lbox: {loss_item[0].asnumpy():.4f}, lobj: "
                  f"{loss_item[1].asnumpy():.4f}, lcls: {loss_item[2].asnumpy():.4f}, "
                  f"cur_lr: [{lr_pg0[i]:.8f}, {lr_pg1[i]:.8f}, {lr_pg2[i]:.8f}], "
                  f"step time: {(time.time() - s_time_step) * 1000:.2f} ms")
        s_time_step = time.time()

        # if (i + 1) % per_epoch_size == 0:
        # if i == 2:
        if False:
            print(f"Epoch {epochs}/{cur_epoch}, epoch time: {(time.time() - s_time_epoch) / 60:.2f} min.")
            if opt.run_eval:
                s_time_eval = time.time()
                if ema:
                    _param_dict = {}
                    for _p in ema.ema_weights:
                        _param_dict[_p.name[len("ema."):]] = _p.data
                else:
                    _param_dict = {}
                    for _p in model.get_parameters():
                        _param_dict[_p.name] = _p.data
                ms.load_param_into_net(eval_model, _param_dict)
                results, maps, times = test(opt.data, None, batch_size * 2, imgsz_test,
                                            model=eval_model,
                                            dataset=eval_dataset,
                                            dataloader=eval_dataloader,
                                            conf_thres=0.001,
                                            iou_thres=0.6,
                                            nms_time_limit=10.0,
                                            single_cls=opt.single_cls,
                                            half_precision=False,
                                            v5_metric=opt.v5_metric)
                fi = fitness(np.array(results).reshape(1, -1))
                if fi > best_fitness:
                    best_fitness = fi
                print(f"Epoch {epochs}/{cur_epoch}, run eval time: {(time.time() - s_time_eval) / 60:.2f} min.")
            s_time_epoch = time.time()

        # save checkpoint
        if (rank % 8 == 0) and ((i + 1) % per_epoch_size == 0):
            # Save Checkpoint
            model_name = os.path.basename(opt.cfg)[:-5] # delete ".yaml"
            ckpt_path = os.path.join(wdir, f"{model_name}_{cur_epoch}.ckpt")
            ms.save_checkpoint(model, ckpt_path)
            if ema:
                params_list = []
                for p in ema.ema_weights:
                    _param_dict = {'name': p.name[len("ema."):], 'data': Tensor(p.data.asnumpy())}
                    params_list.append(_param_dict)

                ema_ckpt_path = os.path.join(wdir, f"EMA_{model_name}_{cur_epoch}.ckpt")
                ms.save_checkpoint(params_list, ema_ckpt_path, append_dict={"updates": ema.updates})

                save_best = opt.run_eval and fi == best_fitness
                best_ckpt_path = os.path.join(wdir, f"EMA_{model_name}_best.ckpt") if save_best else None
                if best_ckpt_path:
                    ms.save_checkpoint(params_list, best_ckpt_path, append_dict={"updates": ema.updates})

    return 0


def load_pretrain(model, opt, ema=None):
    pretrained = opt.weights.endswith('.ckpt')
    if pretrained:
        param_dict = ms.load_checkpoint(opt.weights)
        ms.load_param_into_net(model, param_dict)
        print(f"Pretrain model load from \"{opt.weights}\" success.")
        if ema:
            if opt.ema_weight.endswith('.ckpt'):
                ema.load_param_from_dict(opt.ema_weight)
                print(f"Ema pretrain model load from \"{opt.ema_weight}\" success.")
            else:
                ema.clone_from_model()
                print("ema_weight not exist, default pretrain weight is currently used.")

def create_train_static_shape_fn_gradoperation(model, optimizer, loss_scaler, grad_reducer=None, rank_size=8,
                                               amp_level="O2", overflow_still_update=False, sens=1.0):
    # from mindspore.amp import all_finite # Bugs before MindSpore 1.9.0
    from utils.all_finite import all_finite
    if loss_scaler is None:
        from mindspore.amp import StaticLossScaler
        loss_scaler = StaticLossScaler(sens)
    # Def train func
    use_ota = opt.loss_ota
    use_aux = opt.use_aux
    if use_ota:
        if not use_aux:
            compute_loss = ComputeLossOTA(model)  # init loss class
        else:
            compute_loss = ComputeLossAuxOTA(model) # init loss class
    else:
        if not use_aux:
            compute_loss = ComputeLoss(model)  # init loss class
        else:
            raise NotImplementedError("Not implemented aux loss without ota.")
    ms.amp.auto_mixed_precision(compute_loss, amp_level=amp_level)

    if grad_reducer is None:
        grad_reducer = ops.functional.identity

    def forward_func(x, label, sizes=None):
        x /= 255.0
        if sizes is not None:
            x = ops.interpolate(x, sizes=sizes, coordinate_transformation_mode="asymmetric", mode="bilinear")
        pred = model(x)
        if use_ota:
            loss, loss_items = compute_loss(pred, label, x)
        else:
            loss, loss_items = compute_loss(pred, label)
        loss *= rank_size
        return loss, ops.stop_gradient(loss_items)

    grad_fn = ops.GradOperation(get_by_list=True, sens_param=True)(forward_func, optimizer.parameters)
    sens_value = sens

    @ms.ms_function
    def train_step(x, label, sizes=None, optimizer_update=True):
        loss, loss_items = forward_func(x, label, sizes)
        sens1, sens2 = ops.fill(loss.dtype, loss.shape, sens_value), \
                       ops.fill(loss_items.dtype, loss_items.shape, sens_value)
        grads = grad_fn(x, label, sizes, (sens1, sens2))
        grads = grad_reducer(grads)
        grads = loss_scaler.unscale(grads)
        grads_finite = all_finite(grads)

        if optimizer_update:
            if grads_finite:
                loss = ops.depend(loss, optimizer(grads))
            else:
                if overflow_still_update:
                    loss = ops.depend(loss, optimizer(grads))
                    # print("overflow, still update.")
                else:
                    pass
                    # print("overflow, drop the step.")

        return loss, loss_items, grads, grads_finite

    return train_step


if __name__ == '__main__':
    opt = parse_args("train")
    opt.img_size.extend([opt.img_size[-1]] * (2 - len(opt.img_size)))  # extend to 2 sizes (train, test)
    opt.save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)  # increment run

    context.set_context(mode=context.GRAPH_MODE, device_target=opt.device_target)
    context.set_context(max_call_depth=2000)
    if opt.device_target == "Ascend":
        device_id = int(os.getenv('DEVICE_ID', 0))
        context.set_context(device_id=device_id)
    context.set_context(pynative_synchronize=True)
    # if opt.device_target == "GPU":
    #    context.set_context(enable_graph_kernel=True)
    # Distribute Train
    rank, rank_size, parallel_mode = 0, 1, ParallelMode.STAND_ALONE
    if opt.is_distributed:
        init()
        rank, rank_size, parallel_mode = get_rank(), get_group_size(), ParallelMode.DATA_PARALLEL
    context.set_auto_parallel_context(parallel_mode=parallel_mode, gradients_mean=True, device_num=rank_size)

    opt.total_batch_size = opt.batch_size
    opt.rank_size = rank_size
    opt.rank = rank
    if rank_size > 1:
        assert opt.batch_size % opt.rank_size == 0, '--batch-size must be multiple of device count'
        opt.batch_size = opt.total_batch_size // opt.rank_size

    # Train
    # profiler = Profiler()
    train(opt)
    # profiler.analyse()
