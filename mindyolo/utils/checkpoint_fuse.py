import argparse
import mindspore as ms
from mindspore.train.serialization import save_checkpoint
from mindspore import Tensor

def fuse_checkpoint(opt):
    new_par_dict = {}
    for i in range(opt.start, opt.start + opt.num):
        ckpt_file = opt.base_name + f'_{i}.ckpt'
        par_dict = ms.load_checkpoint(ckpt_file)
        for k in par_dict:
            if k in new_par_dict:
                new_par_dict[k] += par_dict[k].asnumpy()
            else:
                new_par_dict[k] = par_dict[k].asnumpy()

    new_params_list = []
    for k in new_par_dict:
        _param_dict = {'name': k, 'data': Tensor(new_par_dict[k] / opt.num)}
        new_params_list.append(_param_dict)

    ms_ckpt = f"{opt.base_name}_fuse_{opt.start}to{opt.start + opt.num - 1}.ckpt"
    save_checkpoint(new_params_list, ms_ckpt)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='checkpoint_fuse.py')
    parser.add_argument('--num', type=int, default=10, help='fuse checkpoint num')
    parser.add_argument('--start', type=int, default=291, help='Distribute train or not')
    parser.add_argument('--base_name', type=str, default='./yolov7', help='source checkpoint file base')
    opt = parser.parse_args()
    fuse_checkpoint(opt)