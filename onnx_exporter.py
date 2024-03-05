import os
import os.path as osp
import yaml
import time
import argparse

import torch
import torch.nn as nn

from utils import fill_config
from builder import build_from_cfg


def parse_args():
    parser = argparse.ArgumentParser(
            description='A PyTorch project for face recognition.')
    parser.add_argument('--config', 
            help='config files for testing datasets')
    parser.add_argument('--proj_dirs', '--list', nargs='+',
            help='the project directories to be tested')
    parser.add_argument('--batch-size',type=int, default=1, 
            help='biggest batch size')
    parser.add_argument("--dynamic", action="store_true", 
            help="ONNX dynamic axes")
    parser.add_argument('--start_time', 
            help='time to start training')
    args = parser.parse_args()

    return args


def export_onnx(config, batch_size, dynamic):
    # parallel setting
    device_ids = os.environ['CUDA_VISIBLE_DEVICES']
    device_ids = list(range(len(device_ids.split(','))))

    # eval projects one by one
    for proj_dir in config['project']['proj_dirs']:
        # load config
        config_path = osp.join(proj_dir, 'config.yml')
        with open(config_path, 'r') as f:
            test_config = yaml.load(f, yaml.SafeLoader)
    
        # build model
        bkb_net = build_from_cfg(
            test_config['model']['backbone']['net'],
            'model.backbone',
        )

        bkb_net = nn.DataParallel(bkb_net, device_ids=device_ids)
        bkb_net = bkb_net.cuda()
        bkb_net.eval()

        if dynamic:
            # input --> shape(N, 3, h, w), output --> shape(N, feat_size)
            dynamic = {"images": {0: "batch"}, "output": {0: "batch"}}

        # model paths and run test
        model_dir = proj_dir + '/models' #test_config['project']['model_dir']
        save_iters = test_config['project']['save_iters']
        bkb_paths = [
            osp.join(model_dir, 'backbone_{}.pth'.format(save_iter))
            for save_iter in save_iters
        ]

        print(f'Convert {bkb_paths[-1]} to {bkb_paths[-1].replace(".pth", ".onnx")}')
        bkb_net.load_state_dict(torch.load(bkb_paths[-1]))
        dummy_input = torch.randn((batch_size, 3, 112, 112), dtype=torch.float32).cuda()
        bkb_net(dummy_input)
        torch.onnx.export(
                            bkb_net.module, 
                            (dummy_input, ), 
                            f'{bkb_paths[-1].replace(".pth", ".onnx")}',
                            input_names=["images"],
                            output_names=["output"],
                            dynamic_axes=dynamic or None,
                         )


if __name__ == '__main__':
    # get arguments and config
    args = parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f, yaml.SafeLoader)
    config['data'] = fill_config(config['data'])
 
    # override config
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        raise KeyError('Devices IDs have to be specified.'
                'CPU mode is not supported yet')

    if args.proj_dirs:
        config['project']['proj_dirs'] = args.proj_dirs

    export_onnx(config, args.batch_size, args.dynamic)
