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
from mindspore import context
from mindspore.communication.management import init, get_rank, get_group_size
# from mindspore.amp import DynamicLossScaler, all_finite
from mindspore.ops import functional as F
from mindspore.ops import composite as C
from mindspore.ops import operations as P

from network.yolo import Model
from network.common import EMA, ImplicitA, ImplicitM
from network.loss import *
from config.args import get_args_train
from utils.optimizer import get_group_param_yolov7, get_lr_yolov7
from utils.dataset import create_dataloader
from utils.general import increment_path, colorstr, labels_to_class_weights, check_file, check_img_size, all_finite_cpu

id2name = {}

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

def train(hyp, opt):
    ms.set_seed(2)
    save_dir, epochs, batch_size, total_batch_size, weights, rank, freeze = \
        opt.save_dir, opt.epochs, opt.batch_size, opt.total_batch_size, opt.weights, opt.rank, opt.freeze

    with open(opt.data) as f:
        data_dict = yaml.load(f, Loader=yaml.SafeLoader)  # data dict
    nc = 1 if opt.single_cls else int(data_dict['nc'])  # number of classes
    names = ['item'] if opt.single_cls and len(data_dict['names']) != 1 else data_dict['names']  # class names
    assert len(names) == nc, '%g names found for nc=%g dataset in %s' % (len(names), nc, opt.data)  # check

    # Directories
    wdir = os.path.join(save_dir, "weights")
    os.makedirs(wdir, exist_ok=True)
    # Save run settings
    with open(os.path.join(save_dir, "hyp.yaml"), 'w') as f:
        yaml.dump(hyp, f, sort_keys=False)
    with open(os.path.join(save_dir, "opt.yaml"), 'w') as f:
        yaml.dump(vars(opt), f, sort_keys=False)

    # Model
    sync_bn = opt.sync_bn and context.get_context("device_target") == "Ascend" and rank_size > 1
    model = Model(opt.cfg, ch=3, nc=nc, anchors=hyp.get('anchors'), sync_bn=sync_bn, opt=opt)  # create
    if rank % 8 == 0:
        ema_model = Model(opt.cfg, ch=3, nc=nc, anchors=hyp.get('anchors'), sync_bn=sync_bn)
        ema = EMA(model, ema_model)
    else:
        ema = None

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

    # Pretrain
    weights = "./torch2ms_yolov7_torch_init.ckpt"
    opt.ema_weight = "./torch2ms_yolov7_torch_init.ckpt"
    pretrained = weights.endswith('.ckpt')
    if pretrained:
        param_dict = ms.load_checkpoint(weights)
        ms.load_param_into_net(model, param_dict)
        print(f"Pretrain model load from \"{weights}\" success.")
        if ema and opt.ema_weight.endswith('.ckpt'):
            param_dict_ema = ms.load_checkpoint(opt.ema_weight)
            ms.load_param_into_net(ema.ema_model, param_dict_ema)
            if "updates" in param_dict_ema:
                ema.updates = param_dict_ema["updates"]
            else:
                print(f"\"updates\" miss in \"{opt.ema_weight}\"")
            print(f"Ema pretrain model load from \"{opt.ema_weight}\" success.")

    # Freeze
    freeze = [f'model.{x}.' for x in
              (freeze if len(freeze) > 1 else range(freeze[0]))]  # parameter names to freeze (full or partial)
    for n, p in model.parameters_and_names():
        if any(x in n for x in freeze):
            print('freezing %s' % n)
            p.requires_grad = False

    # Image sizes
    gs = max(int(model.stride.asnumpy().max()), 32)  # grid size (max stride)
    nl = model.model[-1].nl  # number of detection layers (used for scaling hyp['obj'])
    imgsz, imgsz_test = [check_img_size(x, gs) for x in opt.img_size]  # verify imgsz are gs-multiples
    train_path = data_dict['train']
    dataloader, dataset, per_epoch_size = create_dataloader(train_path, imgsz, batch_size, gs, opt,
                                                            hyp=hyp, augment=True, cache=opt.cache_images,
                                                            rect=opt.rect, rank_size=opt.rank_size, rank=opt.rank,
                                                            num_parallel_workers=8 if rank_size > 1 else 16,
                                                            image_weights=opt.image_weights, quad=opt.quad,
                                                            prefix=colorstr('train: '))
    mlc = np.concatenate(dataset.labels, 0)[:, 0].max()  # max label class
    assert mlc < nc, 'Label class %g exceeds nc=%g in %s. Possible class labels are 0-%g' % (mlc, nc, opt.data, nc - 1)

    # Optimizer
    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / total_batch_size), 1)  # accumulate loss before optimizing
    hyp['weight_decay'] *= total_batch_size * accumulate / nbs  # scale weight_decay
    print(f"Scaled weight_decay = {hyp['weight_decay']}")
    if "yolov7" in opt.cfg:
        # zhy_test
        #pg0, pg1, pg2 = get_group_param_yolov7(model)
        pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
        for p in model.trainable_params():
            if "weight" in p.name:
                pg1.append(p)
                print(f"pg1: {p.name}")
                continue
            if "bias" in p.name or "beta" in p.name:
                pg2.append(p)
                print(f"pg2: {p.name}")
                continue
            pg0.append(p)
            print(f"pg0: {p.name}")
        lr_pg0, lr_pg1, lr_pg2, momentum_pg, warmup_steps = get_lr_yolov7(opt, hyp, per_epoch_size)
        assert len(momentum_pg) == warmup_steps
        momentum_pg = Tensor(momentum_pg, ms.float32)
    else:
        raise NotImplementedError
    """
    group_params = [{'params': pg0, 'lr': lr_pg0},
                    {'params': pg1, 'lr': lr_pg1}, #'weight_decay': hyp['weight_decay']},
                    {'params': pg2, 'lr': lr_pg2}]
    """
    group_params = [{'params': pg0, 'lr': 0.001},
                    {'params': pg1, 'lr': 0.001}, #'weight_decay': 0.005}, #hyp['weight_decay']},
                    {'params': pg2, 'lr': 0.001}]

    print('Optimizer groups: %g .bias, %g conv.weight, %g other' % (len(pg2), len(pg1), len(pg0)))

    print(f"optimizer loss scale is {opt.ms_optim_loss_scale}")
    #import pdb;pdb.set_trace()
    if opt.optimizer == "sgd":
        #optimizer = nn.SGD(group_params, learning_rate=hyp['lr0'], momentum=hyp['momentum'], nesterov=True,
        #                   loss_scale=opt.ms_optim_loss_scale)
        #optimizer = nn.SGD(group_params, learning_rate=hyp['lr0'])
        optimizer = nn.SGD(model.trainable_params(), learning_rate=0.001)
        #optimizer = nn.SGD(group_params, learning_rate=0.001)
    elif opt.optimizer == "adam":
        optimizer = nn.Adam(group_params, learning_rate=hyp['lr0'], beta1=hyp['momentum'], beta2=0.999,
                            loss_scale=opt.ms_optim_loss_scale)
    else:
        raise NotImplementedError

    # Model parameters
    hyp['box'] *= 3. / nl  # scale to layers
    hyp['cls'] *= nc / 80. * 3. / nl  # scale to classes and layers
    hyp['obj'] *= (imgsz / 640) ** 2 * 3. / nl  # scale to image size and layers
    hyp['label_smoothing'] = opt.label_smoothing
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.gr = 1.0  # iou loss ratio (obj_loss = 1.0 or iou)
    model.class_weights = Tensor(labels_to_class_weights(dataset.labels, nc) * nc)  # attach class weights
    model.names = names
    if ema:
        ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'gr', 'names', 'stride', 'class_weights'])

    # Build train process function
    ms.amp.auto_mixed_precision(model, amp_level="O0")
    grad_reducer = ops.functional.identity
    loss_scaler = None
    train_step = create_train_static_shape_fn_gradoperation(model, optimizer, loss_scaler, grad_reducer,
                                                            amp_level=opt.ms_amp_level,
                                                            overflow_still_update=opt.overflow_still_update,
                                                            sens=opt.ms_grad_sens,
                                                            rank_size=rank_size)

    # train_step_cell = create_train_static_shape_cell(model, optimizer)


    data_loader = dataloader.create_dict_iterator(output_numpy=True, num_epochs=1)
    s_time = time.time()
    accumulate_grads = None
    accumulate_cur_step = 0

    model.set_train(True)
    optimizer.set_train(True)

    for i, data in enumerate(data_loader):
        if i == 10:
            assert False
        if i < warmup_steps:
            # xi = [0, warmup_steps]  # x interp
            # accumulate = max(1, np.interp(i, xi, [1, nbs / total_batch_size]).round())
            if opt.optimizer == "sgd":
                #zhy_test
                #optimizer.momentum = momentum_pg[i]
                pass
                # print("optimizer.momentum: ", optimizer.momentum.asnumpy())

        cur_epoch = (i // per_epoch_size) + 1
        cur_step = (i % per_epoch_size) + 1

        # zhy_test: lock input
        # imgs, labels, paths = data["img"], data["label_out"], data["img_files"]
        imgs, labels = np.load("imgs.npy")[0:1, ...], np.load("labels.npy")[0:1, ...]

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

        # train step
        s_train_time = time.time()
        # import pdb;pdb.set_trace()
        
        # zhy_test
        # 1. original
        loss, loss_item, grad, grads_finite = train_step(imgs, labels, ns, True)
        # loss = train_step_cell(imgs, labels)
        # loss_item = Tensor(np.array([loss[0].asnumpy(), loss[0].asnumpy(), loss[0].asnumpy(), loss[0].asnumpy()]))
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

        #import pdb;pdb.set_trace()
        if ema:
            ema.update()

        #import pdb;pdb.set_trace()

        _p_train_size = ns if ns else imgs.shape[2:]
        print(f"Epoch {epochs}/{cur_epoch}, Step {per_epoch_size}/{cur_step}, size {_p_train_size}, "
              f"fp/bp time cost: {(time.time() - s_train_time) * 1000:.2f} ms")
        print(f"Epoch {epochs}/{cur_epoch}, Step {per_epoch_size}/{cur_step}, size {_p_train_size}, "
              f"loss: {loss.asnumpy():.4f}, lbox: {loss_item[0].asnumpy():.4f}, lobj: "
              f"{loss_item[1].asnumpy():.4f}, lcls: {loss_item[2].asnumpy():.4f}, "
              f"step time: {(time.time() - s_time) * 1000:.2f} ms")
        s_time = time.time()

        if (rank % 8 == 0) and ((i + 1) % per_epoch_size == 0):
            # Save Checkpoint
            model_name = os.path.basename(opt.cfg)[:-5] # delete ".yaml"
            ckpt_path = os.path.join(wdir, f"{model_name}_{cur_epoch}.ckpt")
            ms.save_checkpoint(model, ckpt_path)
            if ema:
                ema_ckpt_path = os.path.join(wdir, f"EMA_{model_name}_{cur_epoch}.ckpt")
                ms.save_checkpoint(ema.ema_model, ema_ckpt_path, append_dict={"updates": ema.updates})

        # TODO: eval every epoch


def create_train_static_shape_cell(model, optimizer, rank_size=1, amp_level="O0"):
    # Def train func
    # use_ota = 'loss_ota' not in hyp or hyp['loss_ota'] == 1
    # if use_ota:
    #     compute_loss = ComputeLossOTA_v2(model)  # init loss class
    # else:
    #     compute_loss = ComputeLoss(model)  # init loss class
    use_ota = 0
    compute_loss = ComputeLossFix(model)

    class Warpper(nn.Cell):
        def __init__(self, model, loss, rank_size=8):
            super(Warpper, self).__init__(auto_prefix=False)
            self.model = model
            self.loss = loss
            self.rank_size = rank_size

        def construct(self, x, label, sizes=None):
            x /= 255.0
            if sizes is not None:
                x = ops.interpolate(x, sizes=sizes, coordinate_transformation_mode="asymmetric", mode="bilinear")
            pred = self.model(x)
            if use_ota:
                loss, loss_items = self.loss(pred, label, x)
            else:
                loss, loss_items = self.loss(pred, label)
            loss *= rank_size
            return loss

    net_with_loss = Warpper(model, compute_loss, rank_size=rank_size)

    # 1. TrainOneStepWithLossScaleCell
    # ms.amp.auto_mixed_precision(net_with_loss, amp_level=amp_level)
    # manager = nn.DynamicLossScaleUpdateCell(loss_scale_value=1024.0, scale_factor=2, scale_window=1000)
    manager = nn.FixedLossScaleUpdateCell(1024.0)
    train_network = nn.TrainOneStepWithLossScaleCell(net_with_loss, optimizer, scale_sense=manager)

    # 2. TrainOneStepCell
    # train_network = nn.TrainOneStepCell(net_with_loss, optimizer, sens=1.0)
    # ms.amp.auto_mixed_precision(train_network, amp_level=amp_level)

    # 3. TrainOneStepWithClipGradientCell
    # train_network = TrainOneStepWithClipGradientCell(net_with_loss, optimizer, sens=1024)
    # ms.amp.auto_mixed_precision(train_network, amp_level=amp_level)

    train_network = train_network.set_train()

    return train_network


def create_train_static_shape_fn_gradoperation(model, optimizer, loss_scaler, grad_reducer=None, amp_level="O2",
                                               overflow_still_update=False, sens=1.0, rank_size=1):
    if loss_scaler is not None:
        print("lossscaler not effective on gradoperation fn.")
    #from mindspore.amp import all_finite
    # Def train func
    use_ota = 'loss_ota' not in hyp or hyp['loss_ota'] == 1
    # if use_ota:
    #     compute_loss = ComputeLossOTA_v2(model)  # init loss class
    # else:
    #     compute_loss = ComputeLoss(model)  # init loss class
    #compute_loss = SimpleComputeLoss()
    #compute_loss = ComputeLoss(model)
    compute_loss = ComputeLossFix(model)
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
        return loss * rank_size, ops.stop_gradient(loss_items)

    grad_fn = ops.GradOperation(get_by_list=True, sens_param=True)(forward_func, optimizer.parameters)
    sens_value = sens

    #@ms.ms_function
    def train_step(x, label, sizes=None, optimizer_update=True):
        loss, loss_items = forward_func(x, label, sizes)
        # sens_value = 1024.0
        sens1, sens2 = ops.fill(loss.dtype, loss.shape, sens_value), \
                       ops.fill(loss_items.dtype, loss_items.shape, sens_value)
        #loss, loss_items = 0, 0
        #sens1, sens2 = ops.fill(ms.float32, (), sens_value), ops.fill(ms.float32, (4,), sens_value)
        #import pdb;pdb.set_trace()
        grads = grad_fn(x, label, sizes, (sens1, sens2))
        grads = grad_reducer(grads)
        # zhy_test
        # grads = loss_scaler.unscale(grads)

        # zhy_test
        for i in range(len(grads)):
            np.save(f"log_hook_result/grad_w_{i}.npy", grads[i].asnumpy())

        #grads_finite = all_finite(grads)
        grads_finite = True

        if optimizer_update:
            if grads_finite:
                loss = ops.depend(loss, optimizer(grads))
            else:
                if overflow_still_update:
                    loss = ops.depend(loss, optimizer(grads))
                    print("overflow, still update.")
                else:
                    print("overflow, drop the step.")

        return loss, loss_items, grads, grads_finite

    return train_step

if __name__ == '__main__':
    opt = get_args_train()
    # opt.hyp = opt.hyp or ('hyp.finetune.yaml' if opt.weights else 'hyp.scratch.yaml')
    opt.data, opt.cfg, opt.hyp = check_file(opt.data), check_file(opt.cfg), check_file(opt.hyp)  # check files
    assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'
    opt.img_size.extend([opt.img_size[-1]] * (2 - len(opt.img_size)))  # extend to 2 sizes (train, test)
    opt.name = 'evolve' if opt.evolve else opt.name
    opt.save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok | opt.evolve)  # increment run

    #ms_mode = context.GRAPH_MODE if opt.ms_mode == "graph" else context.PYNATIVE_MODE
    context.set_context(mode=context.PYNATIVE_MODE, device_target=opt.device_target)
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

    # Hyperparameters
    with open(opt.hyp) as f:
        hyp = yaml.load(f, Loader=yaml.SafeLoader)  # load hyps

    # Train
    if not opt.evolve:
        # profiler = Profiler()
        # check_train(hyp, opt)
        train(hyp, opt)
        # profiler.analyse()
    else:
        raise NotImplementedError("Not support evolve train;")
