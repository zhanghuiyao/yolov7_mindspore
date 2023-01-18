import glob
import os
import time
import json
import datetime
import numpy as np
from pathlib import Path

import mindspore as ms
from mindspore import Tensor, ops

from mindyolo.models.yolo import Model
from mindyolo.utils.general import coco80_to_coco91_class, check_img_size, xyxy2xywh, xywh2xyxy, colorstr, box_iou
from mindyolo.utils.dataset import create_dataloader
from mindyolo.utils.metrics import ConfusionMatrix, non_max_suppression, scale_coords, ap_per_class
from mindyolo.utils.config import parse_args

def test(opt,
         weights=None,
         batch_size=32,
         imgsz=640,
         conf_thres=0.001,
         iou_thres=0.6,  # for NMS
         nms_time_limit=20.0, # for NMS
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
         compute_loss=None,
         half_precision=False,
         is_coco=False,
         v5_metric=False):

    is_coco = opt.dataset_name == "coco"
    nc = 1 if single_cls else int(opt.nc)  # number of classes
    iouv = np.linspace(0.5, 0.95, 10)  # iou vector for mAP@0.5:0.95
    niou = np.prod(iouv.shape)

    # Initialize/load model and set device
    training = model is not None
    if not training:  # called by train.py
        # Directories
        save_dir = os.path.join(opt.project, datetime.datetime.now().strftime('%Y-%m-%d_time_%H_%M_%S'))
        os.makedirs(os.path.join(save_dir, "labels"), exist_ok=False)

        # Load model
        model = Model(opt.cfg, ch=3, nc=nc, sync_bn=False)  # create
        ckpt_path = weights
        param_dict = ms.load_checkpoint(ckpt_path)
        ms.load_param_into_net(model, param_dict)
        print(f"load ckpt from \"{ckpt_path}\" success.")
        gs = max(int(ops.cast(model.stride, ms.float16).max()), 32)  # grid size (max stride)
        imgsz = check_img_size(imgsz, s=gs)  # check img_size

    # Half
    if half_precision:
        ms.amp.auto_mixed_precision(model, amp_level="O2")

    model.set_train(False)

    # Dataloader
    if not training:
        assert opt.task in ('train', 'val'), f"Not support task type {opt.task}"
        data_path = opt.train_set if opt.task == "train" else opt.val_set # path to train/val/test images
        dataloader, dataset, per_epoch_size = create_dataloader(data_path, imgsz, batch_size, gs, opt,
                                                                epoch_size=1, pad=0.5, rect=False,
                                                                num_parallel_workers=8, shuffle=False,
                                                                drop_remainder=False,
                                                                prefix=colorstr(f'{opt.task}: '))
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
    # confusion_matrix = ConfusionMatrix(nc=nc)
    names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
    coco91class = coco80_to_coco91_class()
    p, r, f1, mp, mr, map50, map, t0, t1 = 0., 0., 0., 0., 0., 0., 0., 0., 0.
    loss = np.zeros(3)
    jdict, stats, ap, ap_class = [], [], [], []
    s_time = time.time()
    for batch_i, meta_data in enumerate(data_loader):
        img, targets, paths, shapes = meta_data["img"], meta_data["label_out"],\
                                      meta_data["img_files"], meta_data["shapes"]
        dtype = ms.float16 if half_precision else ms.float32
        img = img.astype(np.float) / 255.0 # 0 - 255 to 0.0 - 1.0
        img_tensor = Tensor(img, dtype)
        targets_tensor = Tensor(targets, dtype) if compute_loss else None
        targets = targets.reshape((-1, 6))
        targets = targets[targets[:, 1] >= 0]
        nb, _, height, width = img.shape  # batch size, channels, height, width

        # Run model
        t = time.time()
        pred_out, train_out = model(img_tensor, augment=augment)  # inference and training outputs
        t0 += time.time() - t

        # Compute loss # metric with common loss
        if compute_loss:
            loss += compute_loss(train_out, targets_tensor)[1][:3].asnumpy()  # box, obj, cls

        # Run NMS
        targets[:, 2:] *= np.array([width, height, width, height], targets.dtype)  # to pixels
        lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
        t = time.time()
        out = pred_out.asnumpy()
        out = non_max_suppression(out, conf_thres=conf_thres, iou_thres=iou_thres, labels=lb, multi_label=True,
                                  time_limit=nms_time_limit)
        t1 += time.time() - t

        # Statistics per image
        for si, pred in enumerate(out):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            path = Path(str(paths[si]))
            seen += 1

            if len(pred) == 0:
                if nl:
                    stats.append((np.zeros((0, niou), dtype=np.bool),
                                  np.zeros(0, dtype=pred.dtype),
                                  np.zeros(0, dtype=pred.dtype),
                                  tcls))
                continue

            # Predictions
            predn = np.copy(pred)
            scale_coords(img[si].shape[1:], predn[:, :4], shapes[si][0, :], shapes[si][1:, :])  # native-space pred

            # Append to text file
            if save_txt:
                gn = np.array(shapes[si][0])[[1, 0, 1, 0]]  # normalization gain whwh
                for *xyxy, conf, cls in predn.tolist():
                    xywh = (xyxy2xywh(np.array(xyxy).reshape((1, 4))) / gn).reshape(-1).tolist()  # normalized xywh
                    line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                    with open(os.path.join(save_dir, 'labels', (path.stem + '.txt')), 'a') as f:
                        f.write(('%g ' * len(line)).rstrip() % line + '\n')

            # Append to pycocotools JSON dictionary
            if save_json:
                # [{"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}, ...
                image_id = int(path.stem) if path.stem.isnumeric() else path.stem
                box = xyxy2xywh(predn[:, :4])  # xywh
                box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
                for p, b in zip(pred.tolist(), box.tolist()):
                    jdict.append({'image_id': image_id,
                                  'category_id': coco91class[int(p[5])] if is_coco else int(p[5]),
                                  'bbox': [round(x, 3) for x in b],
                                  'score': round(p[4], 5)})

            # Assign all predictions as incorrect
            correct = np.zeros((pred.shape[0], niou), dtype=np.bool)
            if nl:
                detected = []  # target indices
                tcls_np = labels[:, 0]

                # target boxes
                tbox = xywh2xyxy(labels[:, 1:5])
                scale_coords(img[si].shape[1:], tbox, shapes[si][0, :], shapes[si][1:, :])  # native-space labels

                # Per target class
                for cls in np.unique(tcls_np):
                    # ti = (cls == tcls_np).nonzero(as_tuple=False).view(-1)  # prediction indices
                    # pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)  # target indices
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

        print(f"Test step {batch_i + 1}/{per_epoch_size}, cost time {time.time() - s_time:.2f}s", flush=True)

        s_time = time.time()

    # Compute statistics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class = ap_per_class(*stats, plot=False, v5_metric=v5_metric, save_dir=save_dir, names=names)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = np.zeros(1)

    # Print results
    pf = '%20s' + '%12i' * 2 + '%12.3g' * 4  # print format
    print(pf % ('all', seen, nt.sum(), mp, mr, map50, map))

    # Print results per class
    if (verbose or (nc < 50 and not training)) and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            print(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

    # Print speeds
    t = tuple(x / seen * 1E3 for x in (t0, t1, t0 + t1)) + (imgsz, imgsz, batch_size)  # tuple
    if not training:
        print('Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g' % t)

    # Save JSON
    if save_json and len(jdict):
        w = Path(weights).stem if weights is not None else ''  # weights
        # anno_json = './coco/annotations/instances_val2017.json'  # annotations json
        anno_json = os.path.join(opt.val_set[:-12], "annotations/instances_val2017.json")
        pred_json = os.path.join(save_dir, f"{w}_predictions.json") # predictions json
        print('\nEvaluating pycocotools mAP... saving %s...' % pred_json)
        with open(pred_json, 'w') as f:
            json.dump(jdict, f)

        try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            anno = COCO(anno_json)  # init annotations api
            pred = anno.loadRes(pred_json)  # init predictions api
            eval = COCOeval(anno, pred, 'bbox')
            if is_coco:
                eval.params.imgIds = [int(Path(x).stem) for x in dataset.img_files]  # image IDs to evaluate
            eval.evaluate()
            eval.accumulate()
            eval.summarize()
            map, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
        except Exception as e:
            print(f'pycocotools unable to run: {e}')

    # Return results
    if not training:
        s = f"\n{len(glob.glob(os.path.join(save_dir, 'labels/*.txt')))} labels saved to " \
            f"{os.path.join(save_dir, 'labels')}" if save_txt else ''
        print(f"Results saved to {save_dir}, {s}")
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]

    model.set_train()
    return (mp, mr, map50, map, *(loss / per_epoch_size).tolist()), maps, t


if __name__ == '__main__':
    opt = parse_args("test")
    print(opt)
    opt.save_json |= (opt.dataset_name == "coco")

    ms.set_context(mode=opt.ms_mode, device_target=opt.device_target)
    ms.set_context(pynative_synchronize=True)

    if opt.task in ('train', 'val', 'test'):  # run normally
        test(opt,
             opt.weights,
             opt.batch_size,
             opt.img_size,
             opt.conf_thres,
             opt.iou_thres,
             opt.nms_time_limit,
             opt.save_json,
             opt.single_cls,
             opt.augment,
             opt.verbose,
             save_txt=opt.save_txt | opt.save_hybrid,
             save_hybrid=opt.save_hybrid,
             save_conf=opt.save_conf,
             half_precision=False,
             v5_metric=opt.v5_metric
             )
    else:
        raise NotImplementedError