import os
import time
import json
import datetime
import numpy as np
from pathlib import Path

from mindyolo.utils.general import coco80_to_coco91_class, xyxy2xywh, xywh2xyxy, colorstr, box_iou
from mindyolo.utils.dataset import create_dataloader
from mindyolo.utils.metrics import non_max_suppression, scale_coords, ap_per_class
from mindyolo.utils.config import parse_args

def sigmoid(x):
    """
    Implements the sigmoid activation in numpy
    """
    return 1 / (1 + np.exp(-x))

def postprocess(opt):
    """
    generate img bin file
    """
    result_path = opt.result_path
    stride, anchors, nc = opt.stride, opt.anchors, opt.nc
    gs = max(max(stride), 32)
    no = nc + 5
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    num_layer = len(anchors)  # number of detection layers
    anchor_grid = np.array(anchors).reshape((num_layer, 1, -1, 1, 1, 2))
    iouv = np.linspace(0.5, 0.95, 10)  # iou vector for mAP@0.5:0.95
    niou = np.prod(iouv.shape)
    is_coco = (opt.dataset_name == "coco")
    save_json = is_coco

    task = 'val'
    dataloader, dataset, per_epoch_size = create_dataloader(opt.val_set, opt.img_size, opt.per_batch_size, gs, opt,
                                                            epoch_size=1, pad=0.5, rect=False,
                                                            num_parallel_workers=8, shuffle=False,
                                                            drop_remainder=False,
                                                            prefix=colorstr(f'{task}: '))
    total_size = dataloader.get_dataset_size()
    assert per_epoch_size == total_size, "total size not equal per epoch size."
    print("Total {} images to postprocess...".format(total_size))
    data_loader = dataloader.create_dict_iterator(output_numpy=True, num_epochs=1)


    seen = 0
    coco91class = coco80_to_coco91_class()
    p, r, f1, mp, mr, map50, map, t0, t1 = 0., 0., 0., 0., 0., 0., 0., 0., 0.
    jdict, stats, ap, ap_class = [], [], [], []
    s_time = time.time()
    for batch_i, meta_data in enumerate(data_loader):
        img, targets, paths, shapes = meta_data["img"], meta_data["label_out"],\
                                      meta_data["img_files"], meta_data["shapes"]
        targets = targets.reshape((-1, 6))
        targets = targets[targets[:, 1] >= 0]
        nb, _, height, width = img.shape  # batch size, channels, height, width
        out_size = [int(opt.img_size // s) for s in opt.stride]
        z = ()
        for _i in range(num_layer):
            ny, nx = out_size[_i], out_size[_i] # (bs,255,20,20)
            file_path = os.path.join(result_path, f"{batch_i}_{_i}.bin")
            out = np.fromfile(file_path, dtype=np.float32).reshape((opt.per_batch_size, na, ny, nx, no))

            xv, yv = np.meshgrid(np.arange(ny), np.arange(nx))
            grid_np = np.stack((xv, yv), 2).reshape((1, 1, ny, nx, 2))
            y = sigmoid(out)
            y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + grid_np) * stride[_i]  # xy
            y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * anchor_grid[_i]  # wh
            z += (y.reshape((nb, -1, no)),)
        pred_out = np.concatenate(z, axis=1)
        out = pred_out

        # Run NMS
        targets[:, 2:] *= np.array([width, height, width, height], targets.dtype)  # to pixels
        t = time.time()
        out = non_max_suppression(out, conf_thres=opt.conf_thres, iou_thres=opt.iou_thres, labels=[], multi_label=True)
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
    save_dir = os.path.join(opt.project, datetime.datetime.now().strftime('%Y-%m-%d_time_%H_%M_%S'))
    os.makedirs(os.path.join(save_dir, "labels"), exist_ok=False)
    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class = ap_per_class(*stats, plot=False, v5_metric=False, save_dir=save_dir)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = np.zeros(1)

    # Print results
    pf = '%20s' + '%12i' * 2 + '%12.3g' * 4  # print format
    print(pf % ('all', seen, nt.sum(), mp, mr, map50, map))

    # Print speeds
    t = tuple(x / seen * 1E3 for x in (t0, t1, t0 + t1)) + (opt.img_size, opt.img_size, opt.per_batch_size)  # tuple
    print('Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g' % t)

    # Save JSON
    if save_json and len(jdict):
        w = 'yolov7'  # weights
        # anno_json = './coco/annotations/instances_val2017.json'  # annotations json
        anno_json = os.path.join(opt.val_set[:-12], "annotations/instances_val2017.json")
        pred_json = os.path.join(save_dir, f"{w}_predictions.json")  # predictions json
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

    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]


if __name__ == '__main__':
    opt = parse_args("export")
    postprocess(opt)