# code modified https://github.com/RapidAI/YOLO2COCO

# -*- encoding: utf-8 -*-
# @File: yolo2coco.py
# @Author: SWHL
# @Contact: liekkaskono@163.com


# YOLO directory structure 1
# This is the structure used by official repository
# parent_folder
# ├── yolov5
# └── datasets
#     └── coco
#         ├── images
#         │   ├── 0001.jpg
#         │   ├── 0002.jpg
#         |   └── ...
#         └── labels
#             ├── 0001.txt
#             ├── 0002.txt
#             └── ...


# YOLO directory structure 2
# This is another structure support by this script
# If you use this structure, you should pass label
# parent_folder
# ├── yolov5
# └── datasets
#     └── coco
#         └── images
#             ├── 0001.jpg
#             ├── 0001.txt
#             ├── 0002.jpg
#             ├── 0002.txt
#             └── ...


import argparse
import json
import time
import warnings
import shutil
import ast
from pathlib import Path

import cv2
from tqdm import tqdm


# ================================================================
# REPLACE THE CLASS NAMES AND ID MAPS IF YOU USE CUSTOM DATASET
# ================================================================
def get_coco_class_names():
    coco_class_names = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
        'hair drier', 'toothbrush'
    ]
    return coco_class_names

# category id corresponding to class names
def get_coco_category_ids():
    coco_category_ids = [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
        35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
        64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90
    ]
    return coco_category_ids
# ================================================================


class YOLO2COCO:
    def __init__(self, data_dir, output_dir,
                 class_names, class_map,
                 mode='val',
                 annotation_only=True):
        self.raw_data_dir = Path(data_dir)
        self.verify_exists(self.raw_data_dir / 'images')
        if not self.verify_exists(self.raw_data_dir / 'labels', strict=False):
            print(f"[INFO] Suppose labels are stored together with images.")
        self.output_dir = Path(output_dir)
        self.mode = mode
        self.annotation_only = annotation_only
        self.save_img_dir = self.output_dir / f'{mode}2017'
        self.anno_dir = self.output_dir / "annotations"
        self.class_names = class_names
        self.class_map = class_map
        self._init_json()

    def __call__(self):
        # Read the image txt.
        mode = self.mode
        txt_path = self.raw_data_dir / f'{mode}2017.txt'
        self.verify_exists(txt_path)
        img_list = self.read_txt(txt_path)

        # Create the directory of saving the new image.
        if not self.annotation_only:
            self.mkdir(self.save_img_dir)

        # Generate json file.
        self.mkdir(self.anno_dir)

        save_json_path = self.anno_dir / f'instances_{mode}2017.json'
        json_data = self.convert(img_list)

        self.write_json(save_json_path, json_data)
        print(f'Successfully convert, detail in {self.output_dir}')

    def convert(self, img_list):
        images, annotations = [], []
        mode = self.mode
        for img_path in tqdm(img_list, desc=mode):
            img_path = Path(img_path)
            img_id = int(img_path.stem) if img_path.stem.isnumeric() else img_path.stem
            image_dict = self.get_image_info(img_path, img_id)
            images.append(image_dict)

            label_path = self.raw_data_dir / 'labels' / f'{mode}2017' / f'{Path(img_path).stem}.txt'
            annotation = self.get_annotation(label_path,
                                             img_id,
                                             image_dict['height'],
                                             image_dict['width'])
            if annotation is not None:
                annotations.extend(annotation)

        json_data = {
            'info': self.info,
            'images': images,
            'licenses': self.licenses,
            'type': self.type,
            'annotations': annotations,
            'categories': self.categories,
        }
        return json_data

    def get_image_info(self, img_path: Path, img_id: int):
        if self.raw_data_dir.as_posix() not in img_path.as_posix():
            img_path = self.raw_data_dir / img_path

        self.verify_exists(img_path)

        new_img_name = f'{img_id:012d}.jpg'
        img_src = cv2.imread(str(img_path))
        save_img_path = None
        if not self.annotation_only:
            save_img_path = self.save_img_dir / new_img_name
            if img_path.suffix.lower() == ".jpg":
                shutil.copyfile(img_path, save_img_path)
            else:
                cv2.imwrite(str(save_img_path), img_src)

        height, width = img_src.shape[:2]
        image_info = {
            'date_captured': self.cur_year,
            'file_name': img_path.name if save_img_path is None else save_img_path.name,
            'id': img_id,
            'height': height,
            'width': width,
        }
        return image_info

    def get_annotation(self, label_path: Path, img_id: int, height, width):
        def get_box_info(vertex_info, height, width):
            cx, cy, w, h = [float(i) for i in vertex_info]

            cx = cx * width
            cy = cy * height
            box_w = w * width
            box_h = h * height

            # left top
            x0 = max(cx - box_w / 2, 0)
            y0 = max(cy - box_h / 2, 0)

            # right bottom
            x1 = min(x0 + box_w, width)
            y1 = min(y0 + box_h, height)

            segmentation = [[x0, y0, x1, y0, x1, y1, x0, y1]]
            segmentation = [list(map(lambda x: round(x, 2), seg)) for seg in segmentation]
            bbox = [x0, y0, box_w, box_h]
            bbox = list(map(lambda x: round(x, 2), bbox))
            area = box_w * box_h
            return segmentation, bbox, area

        if not label_path.exists():
            print(f"[WARNING] Label path {label_path} does not exist.")
            return None

        annotation = []
        label_list = self.read_txt(str(label_path))
        for i, one_line in enumerate(label_list):
            label_info = one_line.split(' ')
            if len(label_info) < 5:
                warnings.warn(
                    f'The {i+1} line of the {label_path} has been corrupted.')
                continue

            category_id, vertex_info = label_info[0], label_info[1:]
            segmentation, bbox, area = get_box_info(vertex_info, height, width)
            annotation.append({
                'segmentation': segmentation,
                'area': area,
                'iscrowd': 0,
                'image_id': img_id,
                'bbox': bbox,
                'category_id': self.class_map[int(category_id)],
                'id': self.annotation_id,
            })
            self.annotation_id += 1
        return annotation

    @staticmethod
    def read_txt(txt_path):
        with open(str(txt_path), 'r', encoding='utf-8') as f:
            data = list(map(lambda x: x.rstrip('\n'), f))
        return data

    @staticmethod
    def verify_exists(file_path, strict=True):
        file_path = Path(file_path)
        if not file_path.exists():
            if strict:
                raise FileNotFoundError(f'[ERROR] The {file_path} is not exists!!!')
            else:
                print(f"[WARNING] The {file_path} is not exists.")
                return False
        return True

    @staticmethod
    def mkdir(dir_path):
        Path(dir_path).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def write_json(json_path, content: dict):
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(content, f, ensure_ascii=False)

    def _get_category(self):
        categories = []
        for i, category in enumerate(self.class_names):
            categories.append({
                'supercategory': category,
                'id': self.class_map[i],
                'name': category,
            })
        return categories

    def _init_json(self):
        self.categories = self._get_category()
        self.type = 'instances'
        self.annotation_id = 1

        self.cur_year = time.strftime('%Y', time.localtime(time.time()))
        self.info = {
            'year': int(self.cur_year),
            'version': '1.0',
            'description': 'For object detection',
            'date_created': self.cur_year,
        }

        self.licenses = [{
            'id': 1,
            'name': 'Apache License v2.0',
            'url': 'https://github.com/RapidAI/YOLO2COCO/LICENSE',
        }]

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Datasets converter from YOLOV5 to COCO')
    parser.add_argument('--data_dir', type=str, default='datasets/YOLOV5', help='Dataset root path')
    parser.add_argument('--output_dir', type=str, default='datasets/YOLOv5', help='Output directory path')
    parser.add_argument('--mode', type=str, default='val', choices=['train', 'val'], help='generate which mode')
    parser.add_argument('--annotation_only', type=ast.literal_eval, default=True, help='Only convert annotations')
    args = parser.parse_args()
    converter = YOLO2COCO(data_dir=args.data_dir, output_dir=args.output_dir,
                          class_names=get_coco_class_names(), class_map=get_coco_category_ids(),
                          mode=args.mode, annotation_only=args.annotation_only)
    converter()