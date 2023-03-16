import copy
from collections import deque

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

from config.args import get_args_train
from network.loss import ComputeLoss, ComputeLossOTA
from network.yolo import Model
from network.common import EMA
from utils.dataset import create_dataloader
from utils.boost import build_train_network
from utils.optimizer import get_group_param, get_lr, YoloMomentum
from utils.general import increment_path, colorstr, labels_to_class_weights, check_file, check_img_size
from test import test


def set_seed(seed=2):
    np.random.seed(seed)
    random.seed(seed)
    ms.set_seed(seed)


# clip grad define ----------------------------------------------------------------------------------
clip_grad = ops.MultitypeFuncGraph("clip_grad")
hyper_map = ops.HyperMap()
GRADIENT_CLIP_TYPE = 1  # 0, ClipByValue; 1, ClipByNorm;
GRADIENT_CLIP_VALUE = 10.0


@clip_grad.register("Number", "Number", "Tensor")
def _clip_grad(clip_type, clip_value, grad):
    """
        Clip gradients.

        Inputs:
            clip_type (int): The way to clip, 0 for 'value', 1 for 'norm'.
            clip_value (float): Specifies how much to clip.
            grad (tuple[Tensor]): Gradients.

        Outputs:
            tuple[Tensor]: clipped gradients.
        """
    if clip_type not in (0, 1):
        return grad
    dt = F.dtype(grad)
    if clip_type == 0:
        new_grad = ops.clip_by_value(grad, F.cast(F.tuple_to_array((-clip_value,)), dt),
                                     F.cast(F.tuple_to_array((clip_value,)), dt))
    else:
        new_grad = nn.ClipByNorm()(grad, F.cast(F.tuple_to_array((clip_value,)), dt))
    return new_grad


# ----------------------------------------------------------------------------------------------------
def detect_overflow(epoch, cur_step, grads):
    for i in range(len(grads)):
        tmp = grads[i].asnumpy()
        if np.isinf(tmp).any() or np.isnan(tmp).any():
            print(f"grad_{i}", flush=True)
            print(f"Epoch: {epoch}, Step: {cur_step} grad_{i} overflow, this step drop. ", flush=True)
            return True
    return False


def load_checkpoint_to_yolo(model, ckpt_path, resume):
    trainable_params = {p.name: p.asnumpy() for p in model.get_parameters()}
    param_dict = ms.load_checkpoint(ckpt_path)
    new_params = {}
    ema_prefix = "ema.ema."
    for k, v in param_dict.items():
        if not k.startswith("model.") and not k.startswith("updates") and not k.startswith(ema_prefix):
            continue

        k = k[len(ema_prefix):] if ema_prefix in k else k
        if k in trainable_params:
            if v.shape != trainable_params[k].shape:
                print(f"[WARNING] Filter checkpoint parameter: {k}", flush=True)
                continue
            new_params[k] = v
        else:
            print(f"[WARNING] Checkpoint parameter: {k} not in model", flush=True)

    ms.load_param_into_net(model, new_params)
    print(f"load ckpt from \"{ckpt_path}\" success.", flush=True)
    resume_epoch = 0
    if resume and 'epoch' in param_dict:
        resume_epoch = int(param_dict['epoch'])
        print(f"[INFO] Resume training from epoch {resume_epoch}", flush=True)
    return resume_epoch


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


def val(opt, model, ema, infer_model, val_dataloader, val_dataset, cur_epoch):
    print("[INFO] Evaluating...", flush=True)
    param_dict = {}
    if opt.ema:
        print("[INFO] ema parameter update", flush=True)
        for p in ema.ema_weights:
            name = p.name[len("ema."):]
            param_dict[name] = p.data
    else:
        for p in model.get_parameters():
            name = p.name
            param_dict[name] = p.data

    ms.load_param_into_net(infer_model, param_dict)
    del param_dict
    infer_model.set_train(False)
    info, _, _ = test(opt.data,
                      opt.weights,
                      opt.batch_size,
                      opt.img_size,
                      opt.conf_thres,
                      opt.iou_thres,
                      opt.save_json,
                      opt.single_cls,
                      opt.augment,
                      opt.verbose,
                      model=infer_model,
                      dataloader=val_dataloader,
                      dataset=val_dataset,
                      save_txt=opt.save_txt | opt.save_hybrid,
                      save_hybrid=opt.save_hybrid,
                      save_conf=opt.save_conf,
                      trace=not opt.no_trace,
                      plots=False,
                      half_precision=False,
                      v5_metric=opt.v5_metric,
                      is_distributed=opt.is_distributed,
                      rank=opt.rank,
                      rank_size=opt.rank_size,
                      opt=opt,
                      cur_epoch=cur_epoch)
    infer_model.set_train(True)
    return info


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


def train(hyp, opt):
    set_seed()
    if opt.enable_modelarts:
        from utils.modelarts import sync_data
        os.makedirs(opt.data_dir, exist_ok=True)
        sync_data(opt.data_url, opt.data_dir)

    save_dir, epochs, batch_size, total_batch_size, weights, rank, freeze = \
        opt.save_dir, opt.epochs, opt.batch_size, opt.total_batch_size, opt.weights, opt.rank, opt.freeze

    with open(opt.data) as f:
        data_dict = yaml.load(f, Loader=yaml.SafeLoader)  # data dict
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
        yaml.dump(hyp, f, sort_keys=False)
    with open(os.path.join(save_dir, "opt.yaml"), 'w') as f:
        yaml.dump(vars(opt), f, sort_keys=False)
    if opt.enable_modelarts:
        sync_data(save_dir, opt.train_url)

    # Model
    sync_bn = opt.sync_bn and context.get_context("device_target") == "Ascend" and rank_size > 1
    model = Model(opt.cfg, ch=3, nc=nc, anchors=hyp.get('anchors'), sync_bn=sync_bn, opt=opt)  # create
    model.to_float(ms.float16)
    ema = EMA(model) if opt.ema else None

    pretrained = weights.endswith('.ckpt')
    resume_epoch = 0
    if pretrained:
        resume_epoch = load_checkpoint_to_yolo(model, weights, opt.resume)
        ema.clone_from_model()
        print("ema_weight not exist, default pretrain weight is currently used.")

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
                                                           rank=rank, rank_size=rank_size,
                                                           num_parallel_workers=4 if rank_size > 1 else 8,
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

    pg0, pg1, pg2 = get_group_param(model)
    lr_pg0, lr_pg1, lr_pg2, momentum_pg, warmup_steps = get_lr(opt, hyp, per_epoch_size)
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

        def is_save_epoch():
            return (cur_epoch >= opt.start_save_epoch) and (cur_epoch % opt.save_interval == 0)

        if opt.save_checkpoint and (rank % 8 == 0) and is_save_epoch():
            # Save Checkpoint
            model_name = os.path.basename(opt.cfg)[:-5]  # delete ".yaml"
            ckpt_path = os.path.join(wdir, f"{model_name}_{cur_epoch}.ckpt")
            ms.save_checkpoint(model, ckpt_path, append_dict={"epoch": cur_epoch})
            ckpt_queue.append(ckpt_path)
            if ema:
                ema_ckpt_path = os.path.join(wdir, f"EMA_{model_name}_{cur_epoch}.ckpt")
                append_dict = {"updates": ema.updates, "epoch": cur_epoch}
                save_ema(ema, ema_ckpt_path, append_dict)
                ema_ckpt_queue.append(ema_ckpt_path)
                print("save ckpt path:", ema_ckpt_path, flush=True)
            if opt.enable_modelarts:
                sync_data(ckpt_path, opt.train_url + "/weights/" + ckpt_path.split("/")[-1])
                if ema:
                    sync_data(ema_ckpt_path, opt.train_url + "/weights/" + ema_ckpt_path.split("/")[-1])

        # Evaluation
        def is_eval_epoch():
            return cur_epoch == opt.eval_start_epoch or \
                   ((cur_epoch >= opt.eval_start_epoch) and (cur_epoch % opt.eval_epoch_interval) == 0)

        if opt.run_eval and is_eval_epoch():
            eval_results = val(opt, model, ema, infer_model, val_dataloader, val_dataset, cur_epoch=cur_epoch)
            mean_ap = eval_results[3]
            if (rank % 8 == 0) and (mean_ap > best_map):
                best_map = mean_ap
                print(f"[INFO] Best result: Best mAP [{best_map}] at epoch [{cur_epoch}]", flush=True)
                # save best checkpoint
                model_name = Path(opt.cfg).stem  # delete ".yaml"
                ckpt_path = os.path.join(wdir, f"{model_name}_best.ckpt")
                ms.save_checkpoint(model, ckpt_path, append_dict={"epoch": cur_epoch})
                if ema:
                    ema_ckpt_path = os.path.join(wdir, f"EMA_{model_name}_best.ckpt")
                    append_dict = {"updates": ema.updates, "epoch": cur_epoch}
                    save_ema(ema, ema_ckpt_path, append_dict)
                if opt.enable_modelarts:
                    sync_data(ckpt_path, opt.train_url + "/weights/" + ckpt_path.split("/")[-1])
                    if ema:
                        sync_data(ema_ckpt_path, opt.train_url + "/weights/" + ema_ckpt_path.split("/")[-1])
                map_str_path = os.path.join(wdir, f"{model_name}_{cur_epoch}_map.txt")
                coco_map_table_str = eval_results[-1]
                with open(map_str_path, 'w') as file:
                    file.write(f"COCO API:\n{coco_map_table_str}\n")
    return 0


if __name__ == '__main__':
    parser = get_args_train()
    opt = parser.parse_args()
    opt.save_json |= opt.data.endswith('coco.yaml')
    # opt.hyp = opt.hyp or ('hyp.finetune.yaml' if opt.weights else 'hyp.scratch.yaml')
    opt.data, opt.cfg, opt.hyp = check_file(opt.data), check_file(opt.cfg), check_file(opt.hyp)  # check files
    assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'
    opt.img_size.extend([opt.img_size[-1]] * (2 - len(opt.img_size)))  # extend to 2 sizes (train, test)
    opt.name = 'evolve' if opt.evolve else opt.name
    opt.save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok | opt.evolve)  # increment run

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
        context.set_auto_parallel_context(parallel_mode=parallel_mode, gradients_mean=True, device_num=rank_size, all_reduce_fusion_config=[10, 70, 130, 190, 250, 310])

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
    profiler = None
    if opt.profiler:
        profiler = Profiler()

    if not opt.evolve:
        print(f"[INFO] OPT: {opt}")
        train(hyp, opt)
    else:
        raise NotImplementedError("Not support evolve train;")

    if opt.profiler:
        profiler.analyse()
