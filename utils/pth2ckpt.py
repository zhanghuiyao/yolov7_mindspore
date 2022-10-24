import torch
from mindspore.train.serialization import save_checkpoint
from mindspore import Tensor


# keys_ms_add = ['model.105.grid_cell.0.param', 'model.105.grid_cell.1.param', 'model.105.grid_cell.2.param']
#
# key_map = {
#     'model.0.bn.running_mean': 'model.0.bn.moving_mean',
#     'model.0.bn.running_var': 'model.0.bn.moving_variance',
#     'model.0.bn.weight': 'model.0.bn.gamma',
#     'model.0.bn.bias': 'model.0.bn.beta',
#     "...": "..."
# }


def pytorch2mindspore():
    """

    Returns:
        object:
    """
    torch_pth = 'yolov7_init.pt'
    new_params_list = []
    par_dict = torch.load(torch_pth, map_location='cpu')
    for k in par_dict.keys():
        new_k = ""
        if ".105." in k:
            new_k = k
        elif "rbr" in k:
            if ".0.weight" in k:
                new_k = k[:-len(".0.weight")] + "_conv.weight"
            elif ".0.bias" in k:
                new_k = k[:-len(".0.bias")] + "_conv.bias"
            elif ".1.weight" in k:
                new_k = k[:-len(".1.weight")] + "_norm.gamma"
            elif ".1.bias" in k:
                new_k = k[:-len(".1.bias")] + "_norm.beta"
            elif ".1.running_mean" in k:
                new_k = k[:-len(".1.running_mean")] + "_norm.moving_mean"
            elif ".1.running_var" in k:
                new_k = k[:-len(".1.running_var")] + "_norm.moving_variance"
        elif "bn" in k:
            if "weight" in k:
                new_k = k[:-len("weight")] + "gamma"
            elif "bias" in k:
                new_k = k[:-len("bias")] + "beta"
            elif "running_mean" in k:
                new_k = k[:-len("running_mean")] + "moving_mean"
            elif "running_var" in k:
                new_k = k[:-len("running_var")] + "moving_variance"
        elif "conv" in k:
            new_k = k

        if new_k == "":
            if not "num_batches_tracked" in k:
                print(f"Convert weight keys \"{k}\" not match.")
        else:
            # new_par_dict[new_k] = Tensor(par_dict[k].numpy())
            _param_dict = {'name': new_k, 'data': Tensor(par_dict[k].numpy())}
            new_params_list.append(_param_dict)


    ms_ckpt = "yolov7_torch2ms_init.ckpt"
    save_checkpoint(new_params_list, ms_ckpt)

    print(f"Convert weight \"{torch_pth}\" to \"{ms_ckpt}\" finish.")

if __name__ == '__main__':
    pytorch2mindspore()