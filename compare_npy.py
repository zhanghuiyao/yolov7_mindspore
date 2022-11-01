import numpy as np

def check(x1_path, x2_path):
    x1 = np.load(x1_path)
    x2 = np.load(x2_path)
    abs_diff = np.abs(x1 - x2).mean()
    relative_diff = np.abs(x1 - x2) / (np.abs(x2) + 1e-6)
    relative_diff = relative_diff.mean() * 100

    print(f"abs_diff: {abs_diff}, relative_diff: {relative_diff:.6f}%")


if __name__ == '__main__':
    x1_path = "/data1/zhy/_YOLO/yolo_mindspore/x.npy"
    x2_path = "/data1/zhy/_YOLO/yolov7_torch_check/x.npy"
    check(x1_path, x2_path)