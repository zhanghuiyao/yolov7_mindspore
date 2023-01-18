import os
import numpy as np

def compare_npy():
    file_list = []
    for _, _, filenames in os.walk("./log_hook_result"):
        for f in filenames:
            if f.startswith("model.") and "-f" in f:
                file_list.append(f)
    file_list.sort()

    for f in file_list:
        path_t = f"../yolov7_official/log_hook_result/{f}"
        path_ms = f"./log_hook_result/{f}"
        if "rbr" in f and "_norm" in f:
            path_t = path_t[:-len("_norm-fi.npy")] + ".1" + path_t[-len("-fi.npy"):]
        elif "rbr" in f and "_conv" in f:
            path_t = path_t[:-len("_conv-fi.npy")] + ".0" + path_t[-len("-fi.npy"):]

        if not os.path.isfile(path_t):
            print(f"file not exist, [{path_t}].")
            continue
        x_ms = np.load(path_ms)
        x_t = np.load(path_t)

        try:
            if (np.array(x_ms.shape) - np.array(x_t.shape)).any():
                print(f"file shape not match, [{f}], ms_shape: {x_ms.shape}, t_shape: {x_t.shape}.")
                continue
            diff_abs = np.abs(x_ms - x_t).mean()
            diff_relative = (np.abs(x_ms - x_t) / (np.abs(x_t) + 1e-6)).mean()
            print(f"file: {f}, abs diff: {diff_abs}, relative diff: {diff_relative}")
        except:
            print("compute diff error.")


if __name__ == '__main__':
    compare_npy()
