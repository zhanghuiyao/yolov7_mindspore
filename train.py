import copy
from collections import deque
import sys
import yaml
import os
import random
import time
import numpy as np
from pathlib import Path
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Parameter
from mindspore.context import ParallelMode
from mindspore import context, Tensor
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.ops import functional as F
from mindspore.profiler.profiling import Profiler

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), "yolov7"))
from config.args import get_args_train
from network.yolo import Model
from network.common import EMA
from network.loss import ComputeLoss, ComputeLossOTA
from utils.boost import build_train_network
from utils.optimizer import get_group_param_yolov7, get_lr_yolov7, YoloMomentum
from utils.dataset import create_dataloader
from utils.general import increment_path, colorstr, labels_to_class_weights, check_file, check_img_size
from .test_v7 import test


def set_seed(seed=2):
    np.random.seed(seed)
    random.seed(seed)
    ms.set_seed(seed)


class Dict(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__


def create_train_network(model, use_ota, compute_loss, ema, optimizer, loss_scaler=None, sens=1.0, rank_size=1):
    class NetworkWithLoss(nn.Cell):
        def __init__(self, model, use_ota, compute_loss, rank_size):
            super(NetworkWithLoss, self).__init__()
            self.model = model
            self.use_ota = use_ota
            self.compute_loss = compute_loss
            self.rank_size = rank_size
            self.lbox_loss = Parameter(Tensor(0.0, ms.float32), requires_grad=False, name="lbox_loss")
            self.lobj_loss = Parameter(Tensor(0.0, ms.float32), requires_grad=False, name="lobj_loss")
            self.lcls_loss = Parameter(Tensor(0.0, ms.float32), requires_grad=False, name="lcls_loss")

        def construct(self, x, label, sizes=None):
            x /= 255.0
            if sizes is not None:
                x = ops.interpolate(x, sizes=sizes, coordinate_transformation_mode="asymmetric", mode="bilinear")
            pred = self.model(x)
            if self.use_ota:
                loss, loss_items = self.compute_loss(pred, label, x)
            else:
                loss, loss_items = self.compute_loss(pred, label)
            loss_items = ops.stop_gradient(loss_items)
            loss *= self.rank_size
            loss = F.depend(loss, ops.assign(self.lbox_loss, loss_items[0]))
            loss = F.depend(loss, ops.assign(self.lobj_loss, loss_items[1]))
            loss = F.depend(loss, ops.assign(self.lcls_loss, loss_items[2]))
            return loss

    print(f"[INFO] rank_size: {rank_size}", flush=True)
    net_with_loss = NetworkWithLoss(model, use_ota, compute_loss, rank_size)
    train_step = build_train_network(network=net_with_loss, ema=ema, optimizer=optimizer, level='O0',
                                     boost_level='O1', amp_loss_scaler=loss_scaler, sens=sens)
    return train_step


def save_ema(ema, ema_ckpt_path, append_dict=None):
    params_list = []
    for p in ema.ema_weights:
        _param_dict = {'name': p.name[len("ema."):], 'data': Tensor(p.data.asnumpy())}
        params_list.append(_param_dict)
    ms.save_checkpoint(params_list, ema_ckpt_path, append_dict=append_dict)


class CheckpointQueue:
    def __init__(self, max_ckpt_num):
        self.max_ckpt_num = max_ckpt_num
        self.ckpt_queue = deque()

    def append(self, ckpt_path):
        self.ckpt_queue.append(ckpt_path)
        if len(self.ckpt_queue) > self.max_ckpt_num:
            ckpt_to_delete = self.ckpt_queue.popleft()
            os.remove(ckpt_to_delete)


def train(opt, data_dict, hyp, fn_dict):
    set_seed()
    fn_dict = Dict(fn_dict)
    if opt.enable_modelarts:
        os.makedirs(opt.data_dir, exist_ok=True)
        fn_dict.sync_data(opt.data_url, opt.data_dir, opt.enable_modelarts)

    save_dir, epochs, batch_size, total_batch_size, weights, rank, freeze = \
        opt.save_dir, opt.epochs, opt.batch_size, opt.total_batch_size, opt.weights, opt.rank, opt.freeze

    nc = 1 if opt.single_cls else int(data_dict['nc'])  # number of classes
    names = ['item'] if opt.single_cls and len(data_dict['names']) != 1 else data_dict['names']  # class names
    assert len(names) == nc, '%g names found for nc=%g dataset in %s' % (len(names), nc, opt.data)  # check

    if opt.enable_modelarts:
        data_dict['train'] = os.path.join(opt.data_dir, data_dict['train'])
        data_dict['val'] = os.path.join(opt.data_dir, data_dict['val'])
        data_dict['test'] = os.path.join(opt.data_dir, data_dict['test'])

    # Directories
    wdir = os.path.join(save_dir, "weights")
    os.makedirs(wdir, exist_ok=True)
    # Save run settings
    with open(os.path.join(save_dir, "hyp.yaml"), 'w') as f:
        yaml.dump(dict(hyp), f, sort_keys=False)
    with open(os.path.join(save_dir, "opt.yaml"), 'w') as f:
        yaml.dump(vars(opt), f, sort_keys=False)
    fn_dict.sync_data(save_dir, opt.train_url, opt.enable_modelarts)

    # Model
    sync_bn = opt.sync_bn and context.get_context("device_target") == "Ascend" and opt.rank_size > 1
    model = Model(opt.cfg, ch=3, nc=nc, anchors=hyp.get('anchors'), sync_bn=sync_bn, opt=opt)  # create
    model.to_float(ms.float16)
    ema = EMA(model) if opt.ema else None
    model, resume_epoch = fn_dict.restore_freeze_fn(model, ema, opt, weights, freeze)

    # Image sizes
    gs = max(int(model.stride.asnumpy().max()), 32)  # grid size (max stride)
    nl = model.model[-1].nl  # number of detection layers (used for scaling hyp['obj'])
    imgsz, imgsz_test = [check_img_size(x, gs) for x in opt.img_size]  # verify imgsz are gs-multiples
    train_path = data_dict['train']
    test_path = data_dict['val']
    train_epoch_size = 1 if opt.optimizer == "thor" else opt.epochs - resume_epoch
    dataloader, dataset, per_epoch_size = create_dataloader(train_path, imgsz, batch_size, gs, opt,
                                                            epoch_size=train_epoch_size,
                                                            hyp=hyp, augment=True, cache=opt.cache_images,
                                                            rect=opt.rect, rank_size=opt.rank_size, rank=opt.rank,
                                                            num_parallel_workers=12,
                                                            image_weights=opt.image_weights, quad=opt.quad,
                                                            prefix=colorstr('train: '), model_train=True)
    if opt.save_checkpoint or opt.run_eval:
        infer_model = copy.deepcopy(model) if opt.ema else model
        rect = False
        val_dataloader, val_dataset, _ = create_dataloader(test_path, imgsz, batch_size, gs, opt,
                                                           epoch_size=1, pad=0.5, rect=rect,
                                                           rank=rank, rank_size=opt.rank_size,
                                                           num_parallel_workers=4 if opt.rank_size > 1 else 8,
                                                           shuffle=False,
                                                           drop_remainder=False,
                                                           prefix=colorstr(f'val: '))

    mlc = np.concatenate(dataset.labels, 0)[:, 0].max()  # max label class
    assert mlc < nc, 'Label class %g exceeds nc=%g in %s. Possible class labels are 0-%g' % (mlc, nc, opt.data, nc - 1)

    # Optimizer
    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / total_batch_size), 1)  # accumulate loss before optimizing
    # accumulate = 1  # accumulate loss before optimizing

    hyp['weight_decay'] *= total_batch_size * accumulate / nbs  # scale weight_decay
    print(f"Scaled weight_decay = {hyp['weight_decay']}")

    pg0, pg1, pg2 = get_group_param_yolov7(model)
    lr_pg0, lr_pg1, lr_pg2, momentum_pg, warmup_steps = get_lr_yolov7(opt, hyp, per_epoch_size)

    group_params = [{'params': pg0, 'lr': lr_pg0},
                    {'params': pg1, 'lr': lr_pg1, 'weight_decay': hyp['weight_decay']},
                    {'params': pg2, 'lr': lr_pg2}]
    print(f"optimizer loss scale is {opt.ms_optim_loss_scale}")
    if opt.optimizer == "sgd":
        optimizer = nn.SGD(group_params, learning_rate=hyp['lr0'], momentum=hyp['momentum'], nesterov=True,
                           loss_scale=opt.ms_optim_loss_scale)
    elif opt.optimizer == "momentum":
        optimizer = YoloMomentum(group_params, learning_rate=hyp['lr0'], momentum=momentum_pg, use_nesterov=True,
                                 loss_scale=opt.ms_optim_loss_scale)
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

    # Build train process function
    # amp
    ms.amp.auto_mixed_precision(model, amp_level=opt.ms_amp_level)
    compute_loss = ComputeLoss(model)  # init loss class
    ms.amp.auto_mixed_precision(compute_loss, amp_level=opt.ms_amp_level)

    if opt.ms_loss_scaler == "dynamic":
        from mindspore.amp import DynamicLossScaler
        loss_scaler = DynamicLossScaler(2 ** 12, 2, 1000)
    elif opt.ms_loss_scaler == "static":
        from mindspore.amp import StaticLossScaler
        loss_scaler = StaticLossScaler(opt.ms_loss_scaler_value)
    else:
        loss_scaler = None

    if opt.ms_strategy == "StaticShape":
        use_ota = 'loss_ota' not in hyp or hyp['loss_ota'] == 1
        compute_loss = ComputeLossOTA(model) if use_ota else ComputeLoss(model)  # init loss class
        train_step = create_train_network(model, use_ota, compute_loss, ema, optimizer, loss_scaler=None,
                                          sens=opt.ms_grad_sens, rank_size=opt.rank_size)
    else:
        raise NotImplementedError

    model.set_train(True)
    optimizer.set_train(True)
    best_map = 0.
    run_profiler_epoch = 2
    ema_ckpt_queue = CheckpointQueue(opt.max_ckpt_num)
    ckpt_queue = CheckpointQueue(opt.max_ckpt_num)

    data_size = dataloader.get_dataset_size()
    jit = True if opt.ms_mode.lower() == "graph" else False
    sink_process = ms.data_sink(train_step, dataloader, steps=data_size * epochs, sink_size=data_size, jit=jit)

    for cur_epoch in range(resume_epoch, epochs):
        cur_epoch = cur_epoch + 1
        start_train_time = time.time()
        loss = sink_process()
        end_train_time = time.time()
        print(f"Epoch {epochs-resume_epoch}/{cur_epoch}, step {data_size}, "
              f"epoch time {((end_train_time - start_train_time) * 1000):.2f} ms, "
              f"step time {((end_train_time - start_train_time) * 1000 / data_size):.2f} ms, "
              f"loss: {loss.asnumpy() / opt.batch_size:.4f}, "
              f"lbox loss: {train_step.network.lbox_loss.asnumpy():.4f}, "
              f"lobj loss: {train_step.network.lobj_loss.asnumpy():.4f}, "
              f"lcls loss: {train_step.network.lcls_loss.asnumpy():.4f}.", flush=True)

        if opt.profiler and (cur_epoch == run_profiler_epoch):
            break

        fn_dict.save_ckpt_fn(model, ema, ckpt_queue, ema_ckpt_queue, save_ema,
                             opt, cur_epoch, rank, wdir, fn_dict.sync_data)
        fn_dict.run_eval_fn(opt, data_dict, hyp, model, ema, infer_model, val_dataloader, val_dataset,
                            cur_epoch, rank, best_map, wdir, fn_dict.sync_data)

    return 0


def context_init(opt):
    ms_mode = context.GRAPH_MODE if opt.ms_mode == "graph" else context.PYNATIVE_MODE
    context.set_context(mode=ms_mode, device_target=opt.device_target, save_graphs=False)
    if opt.device_target == "Ascend":
        device_id = int(os.getenv('DEVICE_ID', 0))
        context.set_context(device_id=device_id)
    # Distribute Train
    rank, rank_size, parallel_mode = 0, 1, ParallelMode.STAND_ALONE
    if opt.is_distributed:
        init()
        rank, rank_size, parallel_mode = get_rank(), get_group_size(), ParallelMode.DATA_PARALLEL
    context.set_auto_parallel_context(parallel_mode=parallel_mode, gradients_mean=True, device_num=rank_size)
    return rank, rank_size


def parse_config(local=False, config_fn=None):
    parse = get_args_train()
    # Hyperparameters
    opt, data_dict, hyp = None, None, None
    if config_fn is not None and config_fn(parse) is not None:
        opt, data_dict, hyp = config_fn(parse)
    # opt.hyp = opt.hyp or ('hyp.finetune.yaml' if opt.weights else 'hyp.scratch.yaml')
    assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'
    opt.is_coco = opt.data.endswith('coco.yaml')
    opt.save_json |= opt.data.endswith('coco.yaml')
    opt.img_size.extend([opt.img_size[-1]] * (2 - len(opt.img_size)))  # extend to 2 sizes (train, test)
    opt.name = 'evolve' if opt.evolve else opt.name
    if local:
        rank, rank_size = context_init(opt)
    else:
        rank = 0 if not opt.is_distributed else get_rank()
        rank_size = 1 if not opt.is_distributed else get_group_size()

    opt.img_size.extend([opt.img_size[-1]] * (2 - len(opt.img_size)))  # extend to 2 sizes (train, test)

    if local:
        opt.name = 'evolve' if opt.evolve else opt.name
        opt.save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok | opt.evolve)  # increment run

    opt.total_batch_size = opt.batch_size
    opt.rank_size = rank_size
    opt.rank = rank
    if rank_size > 1:
        assert opt.batch_size % opt.rank_size == 0, '--batch-size must be multiple of device count'
        opt.batch_size = opt.total_batch_size // opt.rank_size
    return opt, data_dict, hyp
