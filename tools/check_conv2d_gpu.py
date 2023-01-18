import numpy as np
import mindspore as ms
from mindspore import ops, nn, Tensor
import torch
import torch.nn.functional as F

def check_conv2d():

    # 1. diff ms net and single op
    x = np.load("ms.model.6.conv.input.npy")
    weight = np.load("ms.model.6.conv.weight.npy")
    out_ms = ops.Conv2D(64, 3)(Tensor(x, ms.float32), Tensor(weight, ms.float32)).asnumpy()

    # 1.1
    out_ms_net = np.load("ms.model.6.conv.output.npy")
    diff_ms_net_and_single_op_abs = np.abs(out_ms_net - out_ms).mean()
    diff_ms_net_and_single_op_relative = (np.abs(out_ms_net - out_ms) / (np.abs(out_ms) + 1e-6)).mean()
    print("diff ms net and single op: ")
    print(f"abs diff: {diff_ms_net_and_single_op_abs}, relative diff: {diff_ms_net_and_single_op_relative}\n")



    # 2. diff ms signle op and torch single op
    x = np.load("torch.model.6.conv.input.npy")
    weight = np.load("torch.model.6.conv.weight.npy")
    out_ms = ops.Conv2D(64, 3)(Tensor(x, ms.float32), Tensor(weight, ms.float32)).asnumpy()
    print("diff ms signle op and torch single op")

    # 2.1.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    out_t = F.conv2d(torch.Tensor(x).to("cuda"), torch.Tensor(weight).to("cuda")).detach().cpu().numpy()
    diff_abs = np.abs(out_ms - out_t).mean()
    diff_relative = (np.abs(out_ms - out_t) / (np.abs(out_t) + 1e-6)).mean()
    print(f"torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic, torch.backends.cudnn.enabled: "
          f"{torch.backends.cudnn.benchmark}, {torch.backends.cudnn.deterministic}, {torch.backends.cudnn.enabled}")
    print(f"abs diff: {diff_abs}, relative diff: {diff_relative}\n")

    # 2.2.
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    out_t = F.conv2d(torch.Tensor(x).to("cuda"), torch.Tensor(weight).to("cuda")).detach().cpu().numpy()
    diff_abs = np.abs(out_ms - out_t).mean()
    diff_relative = (np.abs(out_ms - out_t) / (np.abs(out_t) + 1e-6)).mean()
    print(f"torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic, torch.backends.cudnn.enabled: "
          f"{torch.backends.cudnn.benchmark}, {torch.backends.cudnn.deterministic}, {torch.backends.cudnn.enabled}")
    print(f"abs diff: {diff_abs}, relative diff: {diff_relative}\n")

    # 2.3.
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    out_t = F.conv2d(torch.Tensor(x).to("cuda"), torch.Tensor(weight).to("cuda")).detach().cpu().numpy()
    diff_abs = np.abs(out_ms - out_t).mean()
    diff_relative = (np.abs(out_ms - out_t) / (np.abs(out_t) + 1e-6)).mean()
    print(f"torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic, torch.backends.cudnn.enabled: "
          f"{torch.backends.cudnn.benchmark}, {torch.backends.cudnn.deterministic}, {torch.backends.cudnn.enabled}")
    print(f"abs diff: {diff_abs}, relative diff: {diff_relative}\n")



if __name__ == '__main__':
    check_conv2d()