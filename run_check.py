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
from mindspore import nn, ops, context, Tensor
from mindspore.amp import DynamicLossScaler, StaticLossScaler, all_finite
from mindspore.context import ParallelMode
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.ops import functional as F

from mindspore import Profiler

from network.yolo import Model
from network.common import ModelEMA
from network.loss import *
from config.args import get_args_train
from utils.optimizer import get_group_param_yolov7, get_lr_yolov7
from utils.dataset import create_dataloader
from utils.general import increment_path, colorstr, labels_to_class_weights, check_file, check_img_size, all_finite_cpu

# check code
def check_train(hyp, opt):
    ms.set_seed(2)
    save_dir, epochs, batch_size, total_batch_size, weights, rank, freeze = \
        opt.save_dir, opt.epochs, opt.batch_size, opt.total_batch_size, opt.weights, opt.rank, opt.freeze

    with open(opt.data) as f:
        data_dict = yaml.load(f, Loader=yaml.SafeLoader)  # data dict
    is_coco = data_dict["dataset_name"] == "coco"
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
    model = Model(opt.cfg, ch=3, nc=nc, anchors=hyp.get('anchors'), opt=opt)  # create
    pretrained = weights.endswith('.ckpt')
    if pretrained:
        param_dict = ms.load_checkpoint(weights)
        ms.load_param_into_net(model, param_dict)

    # Freeze
    freeze = [f'model.{x}.' for x in
              (freeze if len(freeze) > 1 else range(freeze[0]))]  # parameter names to freeze (full or partial)
    for n, p in model.parameters_and_names():
        p.requires_grad = True  # train all layers
        if any(x in n for x in freeze):
            print('freezing %s' % n)
            p.requires_grad = False

    # Image sizes
    gs = max(int(ops.cast(model.stride, ms.float16).max()), 32)  # grid size (max stride)
    nl = model.model[-1].nl  # number of detection layers (used for scaling hyp['obj'])
    imgsz, imgsz_test = [check_img_size(x, gs) for x in opt.img_size]  # verify imgsz are gs-multiples

    # Optimizer
    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / total_batch_size), 1)  # accumulate loss before optimizing
    hyp['weight_decay'] *= total_batch_size * accumulate / nbs  # scale weight_decay
    print(f"Scaled weight_decay = {hyp['weight_decay']}")
    per_epoch_size = 10
    if "yolov7" in opt.cfg:
        pg0, pg1, pg2 = get_group_param_yolov7(model)
        lr_pg0, lr_pg1, lr_pg2, momentum_pg, warmup_steps = get_lr_yolov7(opt, hyp, per_epoch_size)
        assert len(momentum_pg) == warmup_steps
        momentum_pg = Tensor(momentum_pg, ms.float32)
    else:
        raise NotImplementedError
    group_params = [{'params': pg0, 'lr': lr_pg0},
                    {'params': pg1, 'lr': lr_pg1, 'weight_decay': hyp['weight_decay']},
                    {'params': pg2, 'lr': lr_pg2}]

    # if opt.optimizer == "sgd":
    #     optimizer = nn.SGD(group_params, learning_rate=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)
    # elif opt.optimizer == "adam":
    #     optimizer = nn.Adam(group_params, learning_rate=hyp['lr0'], beta1=hyp['momentum'], beta2=0.999)
    # else:
    #     raise NotImplementedError
    optimizer = nn.SGD(model.trainable_params(), learning_rate=0.1, momentum=hyp['momentum'], nesterov=True)

    # TODO: EMA

    # TODO: Resume

    # TODO: SyncBatchNorm

    # TODO: eval durning train

    # Model parameters
    hyp['box'] *= 3. / nl  # scale to layers
    hyp['cls'] *= nc / 80. * 3. / nl  # scale to classes and layers
    hyp['obj'] *= (imgsz / 640) ** 2 * 3. / nl  # scale to image size and layers
    hyp['label_smoothing'] = opt.label_smoothing
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.gr = 1.0  # iou loss ratio (obj_loss = 1.0 or iou)
    # model.class_weights = Tensor(labels_to_class_weights(dataset.labels, nc) * nc)  # attach class weights
    model.names = names

    ckpt_path = "yolov7_torch2ms_init.ckpt"
    param_dict = ms.load_checkpoint(ckpt_path)
    ms.load_param_into_net(model, param_dict)
    print(f"load ckpt from \"{ckpt_path}\" success")
    # import pdb;pdb.set_trace()

    model.set_train(True)
    optimizer.set_train(True)

    if opt.ms_strategy == "StaticCell":
        train_step = create_train_static_shape_cell(model, optimizer)
    else:
        # amp
        ms.amp.auto_mixed_precision(model, amp_level="O2")
        loss_scaler = DynamicLossScaler(2 ** 12, 2, 1000)
        # loss_scaler = StaticLossScaler(1024.0)
        if opt.ms_strategy == "StaticShape":
            train_step = create_train_static_shape_fn(opt, model, optimizer, loss_scaler)
        elif opt.ms_strategy == "MultiShape":
            raise NotImplementedError
        elif opt.ms_strategy == "DynamicShape":
            assert opt.ms_mode == "pynative", f"The dynamic shape function under static graph is under development. Please " \
                                              f"look forward to the subsequent MS version."
            train_step = create_train_dynamic_shape_fn(opt, model, optimizer, loss_scaler)
        else:
            raise NotImplementedError

    s_time = time.time()
    # ms.save_checkpoint(model, "yolov7_init.ckpt")
    # assert 1 ==2

    for i in range(50):
        imgs = Tensor(np.load("imgs.npy"), ms.float16)[:batch_size, ...]
        labels = Tensor(np.load("labels.npy"), ms.float16)[:batch_size, ...]
        print("imgs/labels size: ", imgs.shape, labels.shape)
        if opt.ms_strategy == "StaticCell":
            loss, overflow, scaling_sens = train_step(imgs, labels, None)
            loss_item = loss
            if overflow:
                print(f"Step {i}, overflow, loss scale {scaling_sens.asnumpy()}")
            else:
                print(f"Step {i}, train one step success.")
        else:
            _, loss_item, _, grad_finite = train_step(imgs, labels, None, True)
            loss_scaler.adjust(grad_finite)
            if not grad_finite:
                print(f"Step {i}, overflow, adjust loss scale to {loss_scaler.scale_value.asnumpy()}")
            else:
                print("Train one step success.")
        print(f"step {i}, loss: {loss_item}, cost: {(time.time() - s_time) * 1000.:.2f} ms")
        s_time = time.time()
    print("Train Finish.")


def create_train_static_shape_cell(model, optimizer):
    # Def train func
    use_ota = 'loss_ota' not in hyp or hyp['loss_ota'] == 1
    if use_ota:
        compute_loss = ComputeLossOTA_v2(model)  # init loss class
    else:
        compute_loss = ComputeLoss(model)  # init loss class

    class Warpper(nn.Cell):
        def __init__(self, model, loss):
            super(Warpper, self).__init__()
            self.model = model
            self.loss = loss

        def construct(self, x, label, sizes=None):
            x /= 255.0
            if sizes is not None:
                x = ops.interpolate(x, sizes=sizes, coordinate_transformation_mode="asymmetric", mode="bilinear")
            pred = self.model(x)
            if use_ota:
                loss, loss_items = self.loss(pred, label, x)
            else:
                loss, loss_items = self.loss(pred, label)
            return loss

    net_with_loss = Warpper(model, compute_loss)
    # manager = nn.DynamicLossScaleUpdateCell(loss_scale_value=1024.0, scale_factor=2, scale_window=1000)
    manager = nn.FixedLossScaleUpdateCell(1024.0)
    train_network = nn.TrainOneStepWithLossScaleCell(net_with_loss, optimizer, scale_sense=manager)
    ms.amp.auto_mixed_precision(train_network, amp_level="O2")
    train_network = train_network.set_train(True)

    return train_network

def create_train_static_shape_fn(opt, model, optimizer, loss_scaler):
    # Def train func
    use_ota = 'loss_ota' not in hyp or hyp['loss_ota'] == 1
    if use_ota:
        compute_loss = ComputeLossOTA_v2(model)  # init loss class
    else:
        compute_loss = ComputeLoss(model)  # init loss class
    ms.amp.auto_mixed_precision(compute_loss, amp_level="O2")

    if opt.is_distributed:
        mean = context.get_auto_parallel_context("gradients_mean")
        degree = context.get_auto_parallel_context("device_num")
        grad_reducer = nn.DistributedGradReducer(optimizer.parameters, mean, degree)
    else:
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
        return loss_scaler.scale(loss), loss_items

    grad_fn = ops.value_and_grad(forward_func, grad_position=None, weights=optimizer.parameters, has_aux=True)

    # @ms.ms_function
    def train_step(x, label, sizes=None, optimizer_update=True):
        (loss, loss_items), grads = grad_fn(x, label, sizes)
        grads = grad_reducer(grads)
        unscaled_grads = loss_scaler.unscale(grads)
        import pdb;pdb.set_trace()
        grads_finite = all_finite(unscaled_grads)
        # _ = loss_scaler.adjust(grads_finite)

        # loss = ops.depend(loss, optimizer(unscaled_grads))
        if optimizer_update:
            if grads_finite:
                loss = ops.depend(loss, optimizer(unscaled_grads))
            else:
                print("overflow, drop the step.")

        return loss, loss_items, unscaled_grads, grads_finite

    return train_step

def create_train_dynamic_shape_fn(opt, model, optimizer, loss_scaler):
    # # Def train func
    # if 'loss_ota' not in hyp or hyp['loss_ota'] == 1:
    #     compute_loss = ComputeLossOTA_v1_dynamic(model)  # init loss class
    # else:
    #     compute_loss = ComputeLoss_dynamic(model)  # init loss class
    use_ota = True
    compute_loss = ComputeLossOTA_v1_dynamic(model)
    ms.amp.auto_mixed_precision(compute_loss, amp_level="O2")

    if opt.is_distributed:
        mean = context.get_auto_parallel_context("gradients_mean")
        degree = context.get_auto_parallel_context("device_num")
        grad_reducer = nn.DistributedGradReducer(optimizer.parameters, mean, degree)
    else:
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
        return loss_scaler.scale(loss), loss_items

    grad_fn = ops.value_and_grad(forward_func, grad_position=None, weights=optimizer.parameters, has_aux=True)

    def train_step(x, label, sizes=None, optimizer_update=True):
        (loss, loss_items), grads = grad_fn(x, label, sizes)
        grads = grad_reducer(grads)
        unscaled_grads = loss_scaler.unscale(grads)
        grads_finite = all_finite(unscaled_grads)
        # _ = loss_scaler.adjust(grads_finite)

        if optimizer_update:
            if grads_finite:
                loss = ops.depend(loss, optimizer(unscaled_grads))
            else:
                print("overflow, drop the step.")

        return loss, loss_items, unscaled_grads, grads_finite

    def forward_warpper(x, label, sizes=None, optimizer_update=True):
        loss, loss_items = forward_func(x, label, sizes)
        return loss, loss_items, None, Tensor([True], ms.bool_)

    return forward_warpper


if __name__ == '__main__':
    opt = get_args_train()
    opt.data, opt.cfg, opt.hyp = check_file(opt.data), check_file(opt.cfg), check_file(opt.hyp)  # check files
    opt.img_size.extend([opt.img_size[-1]] * (2 - len(opt.img_size)))  # extend to 2 sizes (train, test)
    opt.name = 'evolve' if opt.evolve else opt.name
    opt.save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok | opt.evolve)  # increment run

    ms_mode = context.GRAPH_MODE if opt.ms_mode == "graph" else context.PYNATIVE_MODE
    context.set_context(mode=ms_mode, device_target=opt.device_target)
    # context.set_context(pynative_synchronize=True)
    # if opt.device_target == "GPU":
    #    context.set_context(enable_graph_kernel=True)
    context.reset_auto_parallel_context()
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
        check_train(hyp, opt)
    else:
        raise NotImplementedError("Not support evolve train;")
