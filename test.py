import io
import sys
import glob
import os
import time
import json
import yaml
import codecs
from pathlib import Path
from threading import Thread
from contextlib import redirect_stdout
import numpy as np
import mindspore as ms
from mindspore.context import ParallelMode
from mindspore import context, Tensor, ops
from mindspore.communication.management import init, get_rank, get_group_size
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), "yolov7"))
from network.yolo import Model
from config.args import get_args_test
from utils.general import coco80_to_coco91_class, check_file, check_img_size, xyxy2xywh, xywh2xyxy, \
    colorstr, box_iou
from utils.dataset import create_dataloader
from utils.metrics import ConfusionMatrix, non_max_suppression, scale_coords, ap_per_class
from utils.plots import plot_study_txt, plot_images, output_to_target

class Dict(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__


def process_batch(detections, labels, iouv):
    """
    Return correct prediction matrix
    Arguments:
        detections (array[N, 6]), x1, y1, x2, y2, conf, class
        labels (array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (array[N, 10]), for 10 IoU levels
    """
    correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(np.bool_)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    correct_class = labels[:, 0:1] == detections[:, 5]
    for i in range(len(iouv)):
        x = np.where((iou >= iouv[i]) & correct_class)  # IoU > threshold and classes match
        if x[0].shape[0]:
            matches = np.concatenate((np.stack(x, 1), iou[x[0], x[1]][:, None]), 1)  # [label, detect, iou]
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                # matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return np.array(correct).astype(np.bool_)


def test(data_dict,
         hyp,
         weights=None,
         batch_size=32,
         imgsz=640,
         conf_thres=0.001,
         iou_thres=0.6,  # for NMS
         save_json=False,
         single_cls=False,
         augment=False,
         verbose=False,
         model=None,
         dataloader=None,
         dataset=None,
         save_dir=Path(''),  # for saving images
         save_txt=False,  # for auto-labelling
         save_hybrid=False,  # for hybrid auto-labelling
         save_conf=False,  # save auto-label confidences
         plots=False,
         compute_loss=None,
         half_precision=False,
         trace=False,
         rect=False,
         is_coco=False,
         v5_metric=False,
         is_distributed=False,
         rank=0,
         rank_size=1,
         opt=None,
         cur_epoch=0,
         fn_dict=None):
    fn_dict = Dict(fn_dict)
    nc = 1 if single_cls else int(data_dict['nc'])  # number of classes
    iouv = np.linspace(0.5, 0.95, 10)  # iou vector for mAP@0.5:0.95
    niou = np.prod(iouv.shape)
    synchronize = fn_dict.Synchronize(rank_size) if is_distributed else None
    is_training = model is not None
    project_dir = os.path.join(opt.project, "eval", f"epoch_{cur_epoch}") if is_training else opt.project
    save_dir = os.path.join(project_dir, f"rank_{rank}") if rank_size > 1 else project_dir
    os.makedirs(os.path.join(save_dir, f"labels"), exist_ok=opt.exist_ok)
    # Initialize/load model and set device

    if model is None:  # called by train.py
        # Load model
        # Hyperparameters
        model = Model(opt.cfg, ch=3, nc=nc, anchors=hyp.get('anchors'), sync_bn=False)  # create
        ckpt_path = weights
        fn_dict.load_checkpoint_to_yolo(model, ckpt_path)
        gs = max(int(ops.cast(model.stride, ms.float16).max()), 32)  # grid size (max stride)
        imgsz = check_img_size(imgsz, s=gs)  # check img_size

    gs = max(int(ops.cast(model.stride, ms.float16).max()), 32)  # grid size (max stride)
    imgsz = imgsz[0] if isinstance(imgsz, list) else imgsz
    imgsz = check_img_size(imgsz, s=gs)  # check img_size

    # Half
    if half_precision:
        model.to_float(ms.float16)

    model.set_train(False)

    # Dataloader
    if dataloader is None or dataset is None:
        print("[INFO] enable rect", rect, flush=True)
        task = opt.task if opt.task in ('train', 'val', 'test') else 'val'  # path to train/val/test images
        dataloader, dataset, per_epoch_size = create_dataloader(data_dict[task], imgsz, batch_size, gs, opt,
                                                                epoch_size=1, pad=0.5, rect=rect,
                                                                rank=rank,
                                                                rank_size=rank_size,
                                                                num_parallel_workers=4 if rank_size > 1 else 8,
                                                                shuffle=False,
                                                                drop_remainder=False,
                                                                prefix=colorstr(f'{task}: '))
        assert per_epoch_size == dataloader.get_dataset_size()
        data_loader = dataloader.create_dict_iterator(output_numpy=True, num_epochs=1)
        print(f"Test create dataset success, epoch size {per_epoch_size}.")
    else:
        assert dataset is not None
        assert dataloader is not None
        per_epoch_size = dataloader.get_dataset_size()
        data_loader = dataloader.create_dict_iterator(output_numpy=True, num_epochs=1)

    if v5_metric:
        print("Testing with YOLOv5 AP metric...")

    seen = 0
    confusion_matrix = ConfusionMatrix(nc=nc)
    names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}

    start_idx = 1
    class_map = fn_dict.coco80_to_coco91_class() if is_coco else list(range(start_idx, 1000 + start_idx))
    p, r, f1, mp, mr, map50, map, t0, t1, t2 = 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.
    map_table_str = ''
    loss = np.zeros(3)
    jdict, stats, ap, ap_class = [], [], [], []
    s_time = time.time()
    for batch_i, meta_data in enumerate(data_loader):
        img, targets, paths, shapes = meta_data["img"], meta_data["label_out"], \
                                      meta_data["img_files"], meta_data["shapes"]
        img = img / 255.0  # 0 - 255 to 0.0 - 1.0
        img_tensor = Tensor.from_numpy(img)
        targets_tensor = Tensor.from_numpy(targets)
        if half_precision:
            img_tensor = ops.cast(img_tensor, ms.float16)
            targets_tensor = ops.cast(targets_tensor, ms.float16)

        targets = targets.reshape((-1, 6))
        targets = targets[targets[:, 1] >= 0]
        nb, _, height, width = img.shape  # batch size, channels, height, width
        data_time = time.time() - s_time
        # Run model
        t = time.time()
        # inference and training outputs
        if compute_loss or not augment:
            pred_out, train_out = model(img_tensor)
        else:
            pred_out, train_out = (model(img_tensor, augment=augment), None)
        infer_time = time.time() - t
        t0 += infer_time

        # Compute loss
        if compute_loss:
            loss += compute_loss(train_out, targets_tensor)[1][:3].asnumpy()  # box, obj, cls

        # NMS
        targets[:, 2:] *= np.array([width, height, width, height], targets.dtype)  # to pixels
        lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
        t = time.time()
        out = fn_dict.non_max_suppression(pred_out.asnumpy(),
                                          conf_thres,
                                          iou_thres,
                                          labels=lb,
                                          multi_label=True,
                                          agnostic=single_cls)
        nms_time = time.time() - t
        t1 += nms_time

        # Metrics
        t = time.time()
        for si, pred in enumerate(out):
            labels = targets[targets[:, 0] == si, 1:]
            nl, npr, shape = labels.shape[0], pred.shape[0], shapes[si][0]  # number of labels, predictions
            tcls = labels[:, 0].tolist() if nl else []  # target class
            if type(paths[si]) is np.ndarray or type(paths[si]) is np.bytes_:
                path = Path(str(codecs.decode(paths[si].tostring()).strip(b'\x00'.decode())))
            else:
                path = Path(paths[si])
            correct = np.zeros((npr, niou)).astype(np.bool_)  # init
            seen += 1

            if npr == 0:
                if nl:
                    stats.append((np.zeros((0, niou), dtype=np.bool),
                                  np.zeros(0, dtype=pred.dtype),
                                  np.zeros(0, dtype=pred.dtype),
                                  tcls))

                continue

            # Predictions
            predn = np.copy(pred)
            predn[:, :4] = scale_coords(img[si].shape[1:], predn[:, :4], shapes[si][0, :], shapes[si][1:, :])  # native-space pred

            # Append to text file
            if save_txt:
                fn_dict.save_one_txt(predn, save_conf, shape, file=os.path.join(save_dir, 'labels', f'{path.stem}.txt'))

            # Append to pycocotools JSON dictionary
            if save_json:
                fn_dict.save_one_json(predn, jdict, path, class_map, is_coco)  # append to COCO-JSON dictionary

            # Assign all predictions as incorrect
            correct = np.zeros((pred.shape[0], niou), dtype=np.bool)
            if nl:
                detected = []  # target indices
                tcls_np = labels[:, 0]

                # target boxes
                tbox = xywh2xyxy(labels[:, 1:5])
                scale_coords(img[si].shape[1:], tbox, shapes[si][0, :], shapes[si][1:, :])  # native-space labels
                if plots:
                    confusion_matrix.process_batch(predn, np.concatenate((labels[:, 0:1], tbox), 1))

                # Per target class
                for cls in np.unique(tcls_np):
                    ti = np.nonzero(cls == tcls_np)[0].reshape(-1) # prediction indices
                    pi = np.nonzero(cls == pred[:, 5])[0].reshape(-1) # target indices

                    # Search for detections
                    if pi.shape[0]:
                        # Prediction to target ious
                        all_ious = box_iou(predn[pi, :4], tbox[ti])
                        ious = all_ious.max(1)  # best ious, indices
                        i = all_ious.argmax(1)

                        # Append detections
                        detected_set = set()
                        for j in (ious > iouv[0]).nonzero()[0]:
                            d = ti[i[j]]  # detected target
                            if d.item() not in detected_set:
                                detected_set.add(d.item())
                                detected.append(d)
                                correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                                if len(detected) == nl:  # all targets already located in image
                                    break

            # Append statistics (correct, conf, pcls, tcls)
            stats.append((correct, pred[:, 4], pred[:, 5], tcls))
        metric_time = time.time() - t
        t2 += metric_time
        # Plot images
        if plots and batch_i < 3:
            f = os.path.join(save_dir, f'test_batch{batch_i}_labels.jpg')  # labels
            Thread(target=plot_images, args=(img, targets, paths, f, names), daemon=True).start()
            f = os.path.join(save_dir, f'test_batch{batch_i}_pred.jpg')  # predictions
            Thread(target=plot_images, args=(img, output_to_target(out), paths, f, names), daemon=True).start()

        print(f"Test step: {batch_i + 1}/{per_epoch_size}: cost time [{(time.time() - s_time) * 1e3:.2f}]ms "
              f"Data time: [{data_time * 1e3:.2f}]ms  Infer time: [{infer_time * 1e3:.2f}]ms  "
              f"NMS time: [{nms_time * 1e3:.2f}]ms  "
              f"Metric time: [{metric_time * 1e3:.2f}]ms", flush=True)
        s_time = time.time()

    # fn_dict.compute_metrics(plots, save_dir, names, nc, seen, stats, verbose, training)

    # Print speeds
    total_time = (t0, t1, t2, t0 + t1 + t2, imgsz, imgsz, batch_size)  # tuple
    t = tuple(x / seen * 1E3 for x in total_time[:4]) + (imgsz, imgsz, batch_size)  # tuple
    # if not training:
    print('Speed: %.1f/%.1f/%.1f/%.1f ms inference/NMS/Metric/total per %gx%g image at batch-size %g' % t)
    print('Total time: %.1f/%.1f/%.1f/%.1f s inference/NMS/Metric/total %gx%g image at batch-size %g' % total_time)

    # Plots
    if plots:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))

    # Save JSON
    if save_json and len(jdict):
        w = Path(weights).stem if weights is not None else ''  # weights
        data_dir = Path(data_dict["val"]).parent
        anno_json = os.path.join(data_dir, "annotations/instances_val2017.json")
        if opt.transfer_format and not os.path.exists(
                anno_json):  # data format transfer if annotations does not exists
            print("[INFO] Transfer annotations from yolo to coco format.")
            transformer = fn_dict.YOLO2COCO(data_dir, output_dir=data_dir,
                                            class_names=data_dict["names"], class_map=class_map,
                                            mode='val', annotation_only=True)
            transformer()
        pred_json = os.path.join(save_dir, f"{w}_predictions_{rank}.json")  # predictions json
        print('\nEvaluating pycocotools mAP... saving %s...' % pred_json)
        with open(pred_json, 'w') as f:
            json.dump(jdict, f)
        sync_tmp_file = os.path.join(project_dir, 'sync_file.tmp')
        if is_distributed:
            if rank == 0:
                print(f"[INFO] Create sync temp file at path {sync_tmp_file}", flush=True)
                os.mknod(sync_tmp_file)
            synchronize()
            # Merge multiple results files
            if rank == 0:
                print("[INFO] Merge detection results...", flush=True)
                pred_json, merged_results = fn_dict.merge_json(project_dir, prefix=w)
        try:
            if rank == 0 and (opt.result_view or opt.recommend_threshold):
                print("[INFO] Start visualization result.", flush=True)
                fn_dict.view_result(anno_json, pred_json, data_dict["val"], score_threshold=None,
                                    recommend_threshold=opt.recommend_threshold)
                print("[INFO] Visualization result completed.", flush=True)
        except Exception as e:
            print(f'[WARNING] Visualization eval result fail: {e}', flush=True)

        try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            if rank == 0:
                print("[INFO] Start evaluating mAP...", flush=True)
                imgIds = [int(Path(x).stem) for x in dataset.img_files] if is_coco else None
                map, map50, map_table_str = fn_dict.coco_eval(anno_json, merged_results if is_distributed else jdict,
                                                              imgIds)
                print("[INFO] Finish evaluating mAP.", flush=True)
                print(f"COCO mAP:\n{map_table_str}", flush=True)
                if os.path.exists(sync_tmp_file):
                    print(f"[INFO] Delete sync temp file at path {sync_tmp_file}", flush=True)
                    os.remove(sync_tmp_file)
            else:
                print(f"[INFO] Waiting for rank [0] device...", flush=True)
                while os.path.exists(sync_tmp_file):
                    time.sleep(1)
                print(f"[INFO] Rank [{rank}] continue executing.", flush=True)
        except Exception as e:
            print(f'pycocotools unable to run: {e}')

    # Return results
    if not is_training:
        s = f"\n{len(glob.glob(os.path.join(save_dir, 'labels/*.txt')))} labels saved to " \
            f"{os.path.join(save_dir, 'labels')}" if save_txt else ''
        print(f"Results saved to {save_dir}, {s}")
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]

    model.set_train()
    return (mp, mr, map50, map, *(loss / per_epoch_size).tolist(), map_table_str), maps, t


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
    parse = get_args_test()
    opt, data_dict, hyp = None, None, None
    if config_fn is not None and config_fn(parse) is not None:
        opt, data_dict, hyp = config_fn(parse)
    opt.save_json |= opt.data.endswith('coco.yaml')
    rank = 0 if not opt.is_distributed else get_rank()
    rank_size = 1 if not opt.is_distributed else get_group_size()
    opt.total_batch_size = opt.batch_size
    opt.rank_size = rank_size
    opt.rank = rank
    if rank_size > 1:
        assert opt.batch_size % opt.rank_size == 0, '--batch-size must be multiple of device count'
        opt.batch_size = opt.total_batch_size // opt.rank_size
    return opt, data_dict, hyp


def run_test(opt, data_dict, hyp, default_fn):
    if opt.task in ('train', 'val', 'test'):  # run normally
        print("opt:", opt)
        info, _, _ = test(data_dict,
                          hyp,
                          opt.weights,
                          opt.batch_size,
                          opt.img_size,
                          opt.conf_thres,
                          opt.iou_thres,
                          opt.save_json,
                          opt.single_cls,
                          opt.augment,
                          opt.verbose,
                          save_txt=opt.save_txt | opt.save_hybrid,
                          save_hybrid=opt.save_hybrid,
                          save_conf=opt.save_conf,
                          trace=not opt.no_trace,
                          rect=opt.rect,
                          is_coco=opt.is_coco,
                          plots=False,
                          half_precision=False,
                          v5_metric=opt.v5_metric,
                          is_distributed=opt.is_distributed,
                          rank=opt.rank,
                          rank_size=opt.rank_size,
                          opt=opt,
                          fn_dict=default_fn)

    elif opt.task == 'speed':  # speed benchmarks
        test(data_dict, hyp, opt.weights, opt.batch_size, opt.img_size, 0.25, 0.45,
             save_json=False, plots=False, half_precision=False, v5_metric=opt.v5_metric)

    elif opt.task == 'study':  # run over a range of settings and save/plot
        # python test.py --task study --data coco.yaml --iou 0.65 --weights yolov5.ckpt
        x = list(range(256, 1536 + 128, 128))  # x axis (image sizes)
        f = f'study_{Path(opt.data).stem}_{Path(opt.weights).stem}.txt'  # filename to save to
        y = []  # y axis
        for i in x:  # img-size
            print(f'\nRunning {f} point {i}...')
            r, _, t = test(data_dict, hyp, opt.weights, opt.batch_size, i, opt.conf_thres, opt.iou_thres, opt.save_json,
                           plots=False, half_precision=False, v5_metric=opt.v5_metric)
            y.append(r + t)  # results and times
        np.savetxt(f, y, fmt='%10.4g')  # save
        os.system('zip -r study.zip study_*.txt')
        plot_study_txt(x=x)  # plot
    return info
