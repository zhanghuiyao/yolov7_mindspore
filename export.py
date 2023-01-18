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
python export.py
"""
import os
import numpy as np

import mindspore as ms
from mindspore import Tensor, export, context

from mindyolo.models.yolo import Model
from mindyolo.utils.config import parse_args

def run_export(opt):
    """
    Export the MINDIR file
    Returns:None
    """

    # set context
    context.set_context(mode=context.GRAPH_MODE, device_target=opt.device_target)
    if opt.device_target == "Ascend":
        device_id = int(os.getenv('DEVICE_ID', '0'))
        context.set_context(device_id=device_id)
    else:
        raise NotImplementedError

    # model init
    nc = int(opt.nc)  # number of classes
    net = Model(opt.cfg, ch=3, nc=nc, sync_bn=False)  # create
    assert isinstance(opt.weights, str) and opt.weights.endswith('.ckpt'), f"opt.weights is {opt.weights}"
    param_dict = ms.load_checkpoint(opt.weights)
    ms.load_param_into_net(net, param_dict)
    print(f"load ckpt from \"{opt.weights}\" success.")
    net.set_train(True)

    # export
    input_arr = Tensor(np.ones([opt.per_batch_size, 3, opt.img_size, opt.img_size]), ms.float32)
    file_name = os.path.basename(opt.config)[:-5] # delete ".yaml"
    export(net, input_arr, file_name=file_name, file_format=opt.file_format)


if __name__ == '__main__':
    opt = parse_args("export")
    run_export(opt)
