# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =======================================================================================
"""
pre-process for inference
"""
import os
import yaml
import numpy as np
from config.args import get_args_310
from utils.dataset import create_dataloader
from utils.general import colorstr

def preprocess(opt):
    """
    generate img bin file
    """
    result_path = opt.output_path
    with open(opt.data) as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)
    with open(opt.hyp) as f:
        hyp = yaml.load(f, Loader=yaml.SafeLoader)  # load hyps
    with open(opt.cfg) as f:
        cfg_ymal = yaml.load(f, Loader=yaml.SafeLoader)  # model dict
    gs = max(max(cfg_ymal['stride']), 32)
    task = 'val'
    dataloader, _, per_epoch_size = create_dataloader(data[task], opt.img_size, opt.per_batch_size, gs, opt,
                                                            epoch_size=1, pad=0.5, rect=False,
                                                            num_parallel_workers=8, shuffle=False,
                                                            drop_remainder=False,
                                                            prefix=colorstr(f'{task}: '))
    total_size = dataloader.get_dataset_size()
    assert per_epoch_size == total_size, "total size not equal per epoch size."
    print("Total {} images to preprocess...".format(total_size))
    data_loader = dataloader.create_dict_iterator(output_numpy=True, num_epochs=1)
    img_path = os.path.join(result_path, 'img_data')
    if not os.path.exists(img_path):
        os.makedirs(img_path)

    for i, meta_data in enumerate(data_loader):
        img = meta_data["img"].astype(np.float32)
        img /= 255.0
        file_name = f"{i}.bin"
        img_file_path = os.path.join(img_path, file_name)
        img.tofile(img_file_path)

    print("img bin file generate finished, in %s" % img_path)


if __name__ == '__main__':
    opt = get_args_310()
    preprocess(opt)
