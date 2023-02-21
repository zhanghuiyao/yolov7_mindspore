# Contents test

- [YOLOv7 Description](#YOLOv7-description)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
    - [Training Process](#training-process)
        - [Training](#training)
    - [Evaluation Process](#evaluation-process)
        - [Evaluation](#evaluation)
    - [Convert Process](#convert-process)
        - [Convert](#convert)
- [Model Description](#model-description)
    - [Performance](#performance)
        - [Train Performance](#evaluation-performance)
        - [Evaluation Performance](#inference-performance)
- [ModelZoo Homepage](#modelzoo-homepage)

# [YOLOv7 Description](#contents)

YOLOv7 surpasses all known object detectors in both speed and accuracy in the range from 5 FPS to 160 FPS and has the highest accuracy 56.8% AP among all known real-time object detectors with 30 FPS or higher on GPU V100. YOLOv7-E6 object detector (56 FPS V100, 55.9% AP) outperforms both transformer-based detector SWIN-L Cascade-Mask R-CNN (9.2 FPS A100, 53.9% AP) by 509% in speed and 2% in accuracy, and convolutional-based detector ConvNeXt-XL Cascade-Mask R-CNN (8.6 FPS A100, 55.2% AP) by 551% in speed and 0.7% AP in accuracy, as well as YOLOv7 outperforms: YOLOR, YOLOX, Scaled-YOLOv4, YOLOv5, DETR, Deformable DETR, DINO-5scale-R50, ViT-Adapter-B and many other object detectors in speed and accuracy. Moreover, we train YOLOv7 only on MS COCO dataset from scratch without using any other datasets or pre-trained weights. 

[Paper](https://arxiv.org/abs/2207.02696):

Chien-Yao Wang, Alexey Bochkovskiy, Hong-Yuan Mark Liao. YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors. arXiv preprint arXiv:2207.02696, 2022.

# [Dataset](#contents)

Dataset used: [COCO2017](https://cocodataset.org/#download)
Download MS COCO dataset images ([train](http://images.cocodataset.org/zips/train2017.zip), [val](http://images.cocodataset.org/zips/val2017.zip), [test](http://images.cocodataset.org/zips/test2017.zip)) and [labels](https://github.com/ultralytics/yolov5/releases/download/v1.0/coco2017labels-segments.zip).

- Prepare Dataset
  - Use the following script to download or manually refer to the above link to download

  ```shell
  bash scripts/get_coco.sh
  ```

  - The directory structure is as follows, the name of directory and file is user define:

      ```text
          ├── dataset
              ├── coco
                  ├── annotations
                  │   ├─ instances_train2017.json
                  │   └─ instances_val2017.json
                  ├── images
                  │   ├─train2017
                  │   │   ├─picture1.jpg
                  │   │   ├─ ...
                  │   │   └─picturen.jpg
                  │   └─ val2017
                  │       ├─picture1.jpg
                  │       ├─ ...
                  │       └─picturen.jpg
                  ├── labels
                  │   ├─train2017
                  │   │   ├─label1.txt
                  │   │   ├─ ...
                  │   │   └─labeln.txt
                  │   └─ val2017
                  │       ├─label1.txt
                  │       ├─ ...
                  │       └─labeln.txt
                  ├── train2017.txt
                  ├── val2017.txt
                  └── test-dev2017.txt
      ```

we suggest user to use MS COCO dataset to experience our model,
other datasets need to use the same format as MS COCO.

# [Environment Requirements](#contents)

- Hardware（Ascend）
    - Prepare hardware environment with Ascend processor.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/en/master/index.html)

# [Quick Start](#contents)

- After installing MindSpore via the official website, you can start training and evaluation as follows:
- Prepare the hccl_8p.json files, before run network.
    - Genatating hccl_8p.json, Run the script of [hccl_tools.py](https://gitee.com/mindspore/models/blob/master/utils/hccl_tools/hccl_tools.py).
      The following parameter "[0-8)" indicates that the hccl_8p.json file of cards 0 to 7 is generated.

      ```
      python hccl_tools.py --device_num "[0,8)"
      ```
- Specify the dataset path in config/data/coco.yaml

- Run on local

  ```text
  #run training example(1p) by python command
  python train.py \
      --cfg=./config/network_yolov7/yolov7.yaml \
      --data=./config/data/coco.yaml \
      --hyp=./config/data/hyp.scratch.p5.yaml \
      --recompute=True \
      --recompute_layers=5 \
      --batch_size=16 \
      --epochs=300 > log.txt 2>&1 &

  # standalone training example(1p) by shell script (Training with a single scale)
  bash run_standalone_train_ascend.sh 0

  # For Ascend device, distributed training example(8p) by shell script (Training with multi scale)
  bash run_distribute_train_ascend.sh /path_to/hccl_8p.json

  # run evaluation by python command
  python test.py \
      --weights=/path_to/yolov7.ckpt \
      --img_size=640 > log.txt 2>&1 &

  # run evaluation by shell script
  bash run_standalone_test_ascend.sh /path_to/yolov7.ckpt 0
  ```

- Run on [ModelArts](https://support.huaweicloud.com/modelarts/)

  ```python
  # Run with modelarts
  # (1) Perform a or b.
  #       a. Set the default value of the "enable_modelarts" to "True" on args.py file.
  #          Set the default value of the "data_dir" to "/cache/data/" on args.py file.
  #          Set the default value of the "data_url" to "s3://dir_to_your_dataset/" on args.py file. (notes: No setting is required on the openi platform.)
  #          Set the default value of the "train_url" to "s3://dir_to_your_output/" on args.py file. (notes: No setting is required on the openi platform.)
  #          Set other parameters on args.py file you need.
  #       b. Add "enable_modelarts=True" on the website UI interface.
  #          Add "data_dir=/cache/data/" on the website UI interface.
  #          Add "data_url" and "train_url" on the website UI interface. (notes: No setting is required on the openi platform.)
  #          Add other parameters on the website UI interface.
  # (2) Upload or copy your trained model to S3 bucket. (notes: No setting is required when train.)
  # (3) Upload a zip dataset to S3 bucket. (notes: you could also upload the origin dataset, but it can be so slow.)
  # (4) Set the code directory to "/path/yolov7" on the website UI interface.
  # (5) Set the startup file to "train.py" or "test.py" on the website UI interface.
  # (6) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface. (notes: No setting is required on the openi platform.)
  # (7) Create your job.
  #
  ```

# [Script Description](#contents)

## [Script and Sample Code](#contents)

```text
└─yolov7
  ├─README.md
  ├─scripts
    ├─run_distribute_train_ascend.sh         # launch distributed training(8p) in ascend
    ├─run_standalone_train_ascend.sh         # launch standalone training(1p) in ascend
    └─run_standalone_test_ascend.sh          # launch evaluating in ascend
  ├─ascend310_infer
    ├─inc
        ├─utils.h
    ├─src
        ├─main.cc
        ├─utils.cc
    ├─build.sh
    ├─CMakeLists.txt
  ├─config
    ├─data
        ├─coco.yaml
        ├─hyp.scratch.p5.yaml
    ├─network_yolov7
        ├─yolov7.yaml
    ├─args.py
  ├─network
    ├─__init__.py
    ├─common.py
    ├─loss.py
    ├─yolo.py
  ├─utils
    ├─__init__.py
    ├─all_finite.py
    ├─augumentation.py
    ├─autoanchor.py
    ├─callback.py
    ├─checkpoint_fuse.py
    ├─dataset.py
    ├─general.py
    ├─metrics.py
    ├─modelarts.py
    ├─optimizer.py
    ├─plots.py
    ├─pth2ckpt.py
  ├─scripts
    ├─get_coco.sh
    ├─run_distribute_train_ascend.sh
    ├─run_standalone_train_ascend.sh
    ├─run_standalone_test_ascend.sh
    ├─run_infer_310.sh
  ├─postprocess.py
  ├─preprocess.py
  ├─export.py
  ├─test.py
  └─train.py
```

## [Script Parameters](#contents)

Major parameters config/hyp.scratch.p5.yaml as follows:

Major parameters config/args.py as follows:

```text
optional arguments:
  --device_target       device where the code will be implemented: "Ascend", default is "Ascend"
  --ms_strategy         train strategy, default is "StaticShape".
  --ms_amp_level        amp level, default O0.
  --batch_size BATCH_SIZE
                        Batch size for Training. Default: 128.
  --ms_loss_scaler      train loss scaler, default is "static".
  --weights             The ckpt file of YOLOv7, which used to fine tune. default is "".
  --cfg                 model yaml path
  --data                data yaml path
  --hyp                 hyperparameters yaml path
```

## [Training Process](#contents)

### Training

For Ascend device, standalone training example(1p) by shell script

```bash
bash run_standalone_train.sh 0
```

```text
python train.py \
  --cfg=./config/network_yolov7/yolov7.yaml \
  --data=./config/data/coco.yaml \
  --hyp=./config/data/hyp.scratch.p5.yaml \
  --recompute=True \
  --recompute_layers=5 \
  --batch_size=16 \
  --epochs=300 > log.txt 2>&1 &
```

The python command above will run in the background, you can view the results through the file log.txt.

After training, you'll get some checkpoint files under the outputs folder by default.

### Distributed Training

For Ascend device, distributed training example(8p) by shell script

```bash
bash run_distribute_train_ascend.sh /path_to/hccl_8p.json
```

The above shell script will run distribute training in the background. You can view the results through the file train_parallel[X]/log.txt. The loss value will be achieved as follows:

```text
# distribute training result(8p, dynamic shape)
...
Epoch 300/1, Step 924/800, size (640, 640), loss: 23.1380, lbox: 0.0639, lobj: 0.0299, lcls: 0.0870, cur_lr: [0.00043236, 0.00043236, 0.07160841], step time: 1225.23 ms
Epoch 300/1, Step 924/801, size (640, 640), loss: 23.2474, lbox: 0.0634, lobj: 0.0321, lcls: 0.0861, cur_lr: [0.00043290, 0.00043290, 0.07157287], step time: 1278.61 ms
Epoch 300/1, Step 924/802, size (640, 640), loss: 20.8814, lbox: 0.0652, lobj: 0.0255, lcls: 0.0725, cur_lr: [0.00043344, 0.00043344, 0.07153734], step time: 1291.87 ms
Epoch 300/1, Step 924/803, size (640, 640), loss: 22.0405, lbox: 0.0646, lobj: 0.0225, lcls: 0.0851, cur_lr: [0.00043398, 0.00043398, 0.07150180], step time: 1310.95 ms
Epoch 300/1, Step 924/804, size (640, 640), loss: 22.1711, lbox: 0.0634, lobj: 0.0269, lcls: 0.0828, cur_lr: [0.00043452, 0.00043452, 0.07146627], step time: 1261.18 ms
Epoch 300/1, Step 924/805, size (640, 640), loss: 22.3260, lbox: 0.0650, lobj: 0.0303, lcls: 0.0791, cur_lr: [0.00043506, 0.00043506, 0.07143074], step time: 1266.44 ms
Epoch 300/1, Step 924/806, size (640, 640), loss: 21.9784, lbox: 0.0654, lobj: 0.0239, lcls: 0.0824, cur_lr: [0.00043561, 0.00043561, 0.07139520], step time: 1274.14 ms
Epoch 300/1, Step 924/807, size (640, 640), loss: 22.1859, lbox: 0.0635, lobj: 0.0241, lcls: 0.0858, cur_lr: [0.00043615, 0.00043615, 0.07135967], step time: 1251.04 ms
Epoch 300/1, Step 924/808, size (640, 640), loss: 22.4195, lbox: 0.0658, lobj: 0.0246, lcls: 0.0847, cur_lr: [0.00043669, 0.00043669, 0.07132413], step time: 1266.10 ms
Epoch 300/1, Step 924/809, size (640, 640), loss: 23.9537, lbox: 0.0656, lobj: 0.0349, lcls: 0.0867, cur_lr: [0.00043723, 0.00043723, 0.07128860], step time: 1261.55 ms
Epoch 300/1, Step 924/810, size (640, 640), loss: 21.0179, lbox: 0.0643, lobj: 0.0221, lcls: 0.0778, cur_lr: [0.00043777, 0.00043777, 0.07125307], step time: 1280.73 ms
Epoch 300/1, Step 924/811, size (640, 640), loss: 22.2355, lbox: 0.0600, lobj: 0.0308, lcls: 0.0829, cur_lr: [0.00043831, 0.00043831, 0.07121753], step time: 1254.24 ms
...
```

## [Evaluation Process](#contents)

### Weight average

After training, you can average the weight of the saved checkpoint to obtain higher accuracy.

```python
# Average the weight of the last 80 checkpoints, fuse epoch 221 to 300 checkpoints.
python utils/checkpoint_fuse.py --num 80 --start 221 --base_name /path_to_weights/yolov7
```

### Valid

```shell
python test.py \
  --weights=/path_to/yolov7.ckpt \
  --img_size=640 > log.txt 2>&1 &
OR
bash run_standalone_test_ascend.sh /path_to/yolov7.ckpt 0
```

The above python command will run in the background. You can view the results through the file "log.txt". The mAP of the test dataset will be as follows:

```text
# log.txt
# =============coco eval reulst=========
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.511
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.694
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.555
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.349
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.558
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.660
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.382
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.634
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.684
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.531
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.738
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.824
```

## [Convert Process](#contents)

### Convert

If you want to infer the network on Ascend 310, you should convert the model to MINDIR:

```python
python export.py --weights [CKPT_PATH] --file_format [FILE_FORMAT]
```

The weights parameter is required, `FILE_FORMAT` should be in ["AIR", "ONNX", "MINDIR"]

## [Inference Process](#contents)

**Before inference, please refer to [MindSpore Inference with C++ Deployment Guide](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README.md) to set environment variables.**

### Usage

Before performing inference, the mindir file must be exported by export script on the 910 environment.
Current batch_Size can only be set to 1.

```shell
# Ascend310 inference
bash run_infer_310.sh [MINDIR_PATH] [DEVICE_ID]
```

### result

Inference result is saved in current path, you can find result like this in acc.log file.

```text
# =============coco eval reulst=========
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.511
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.694
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.555
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.349
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.558
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.660
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.381
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.634
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.684
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.531
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.738
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.824
```

# [Model Description](#contents)

## [Performance](#contents)

### Train Performance

| Parameters                 | YOLOv7                                                                |
| -------------------------- |-----------------------------------------------------------------------|
| Resource                   | Ascend 910; CPU 2.60GHz, 192cores; Memory, 755G; System, Euleros 2.8; |
| uploaded Date              | 12/06/2022 (month/day/year)                                           |
| MindSpore Version          | 1.9.0                                                                 |
| Dataset                    | coco2017                                                              |
| Training Parameters        | epoch=300, batch_size=128, lrf=0.01, warmup_epochs=20                 |
| Optimizer                  | Momentum                                                              |
| Loss Function              | Sigmoid Cross Entropy with logits, Ciou Loss                          |
| outputs                    | boxes and labels                                                      |
| Loss                       | 5-10                                                                  |
| Speed                      | 8p 98FPS                                                              |
| Total time                 | 100h                                                                  |
| Checkpoint for Fine tuning | about 150M (.ckpt file)                                               |
| Scripts                    | <https://gitee.com/mindspore/models/tree/master/official/cv/YOLOv7>   |

### Evaluation Performance

| Parameters                 | YOLOv7                                     |
| -------------------------- |--------------------------------------------|
| Resource                   | Ascend 910; CPU 2.60GHz, 192cores; Memory, 755G |
| uploaded Date              | 12/06/2022 (month/day/year)                |
| MindSpore Version          | 1.9.0                                      |
| Dataset                    | coco2017                                   |
| batch_size                 | 1                                          |
| outputs                    | box position and sorces, and probability   |
| Accuracy                   | mAP: 51.1%                                 |
| Model for inference        | about 150M (.ckpt file)                    |

# [Description of Random Situation](#contents)

In train.py, we set the seed inside ```set_seed``` function.

# [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/models).
