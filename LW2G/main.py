import os.path
import sys
import argparse
import datetime
import random
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json

from pathlib import Path

from timm.models import create_model
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer

from datasets import build_continual_dataloader

import utils
import warnings


warnings.filterwarnings('ignore', 'Argument interpolation should be of type InterpolationMode instead of int')

# want to save everything printed to outfile
class Logger(object):
    def __init__(self, name):
        self.terminal = sys.stdout
        self.log = open(name, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        self.log.flush()


def get_args(debug_mode=0):
    parser = argparse.ArgumentParser('DualPrompt training and evaluation configs')
    config = parser.parse_known_args()
    if debug_mode == 0:
        config = config[-1][0]
    else:
        # config = 'cifar100_sprompt_5e' # method's name
        config = 'omnibenchmark_sprompt_5e'
        
    print(config)
    subparser = parser.add_subparsers(dest='subparser_name')

    if config == 'cifar100_hideprompt_5e':
        from configs.cifar100_hideprompt_5e import get_args_parser
        config_parser = subparser.add_parser('cifar100_hideprompt_5e', help='Split-CIFAR100 HiDe-Prompt configs')
    elif config == 'imr_hideprompt_5e':
        from configs.imr_hideprompt_5e import get_args_parser
        config_parser = subparser.add_parser('imr_hideprompt_5e', help='Split-ImageNet-R HiDe-Prompt configs')
    elif config == 'cub_hideprompt_5e':
        from configs.cub_hideprompt_5e import get_args_parser
        config_parser = subparser.add_parser('cub_hideprompt_5e', help='Split-CUB HiDe-Prompt configs')
    elif config == 'cifar100_dualprompt':
        from configs.cifar100_dualprompt import get_args_parser
        config_parser = subparser.add_parser('cifar100_dualprompt', help='Split-CIFAR100 dual-prompt configs')
    elif config == 'imr_dualprompt':
        from configs.imr_dualprompt import get_args_parser
        config_parser = subparser.add_parser('imr_dualprompt', help='Split-ImageNet-R dual-prompt configs')
    elif config == 'cub_dualprompt':
        from configs.cub_dualprompt import get_args_parser
        config_parser = subparser.add_parser('cub_dualprompt', help='Split-CUB dual-prompt configs')
    elif config == 'cifar100_sprompt_5e':
        from configs.cifar100_sprompt_5e import get_args_parser
        config_parser = subparser.add_parser('cifar100_sprompt_5e', help='Split-CIFAR100 s-prompt configs')
    elif config == 'imr_sprompt_5e':
        from configs.imr_sprompt_5e import get_args_parser
        config_parser = subparser.add_parser('imr_sprompt_5e', help='Split-ImageNet-R s-prompt configs')
    elif config == 'cub_sprompt_5e':
        from configs.cub_sprompt_5e import get_args_parser
        config_parser = subparser.add_parser('cub_sprompt_5e', help='Split-CUB s-prompt configs')
    elif config == 'cifar100_l2p':
        from configs.cifar100_l2p import get_args_parser
        config_parser = subparser.add_parser('cifar100_l2p', help='Split-CIFAR100 l2p configs')
    elif config == 'imr_l2p':
        from configs.imr_l2p import get_args_parser
        config_parser = subparser.add_parser('imr_l2p', help='Split-ImageNet-R l2p configs')
    elif config == 'cub_l2p':
        from configs.cub_l2p import get_args_parser
        config_parser = subparser.add_parser('cub_l2p', help='Split-CUB l2p configs')
    elif config == 'cifar100_hidelora':
        from configs.cifar100_hidelora import get_args_parser
        config_parser = subparser.add_parser('cifar100_hidelora', help='Split-CIFAR100 hidelora configs')
    elif config == 'imr_hidelora':
        from configs.imr_hidelora import get_args_parser
        config_parser = subparser.add_parser('imr_hidelora', help='Split-ImageNet-R hidelora configs')
    elif config == 'cifar100_continual_lora':
        from configs.cifar100_continual_lora import get_args_parser
        config_parser = subparser.add_parser('cifar100_continual_lora', help='Split-CIFAR100 continual lora configs')
    elif config == 'imr_continual_lora':
        from configs.imr_continual_lora import get_args_parser
        config_parser = subparser.add_parser('imr_continual_lora', help='Split-ImageNet-R continual lora configs')
    
    elif config == 'omnibenchmark_sprompt_5e':
        from configs.omnibenchmark_sprompt_5e import get_args_parser
        config_parser = subparser.add_parser('omnibenchmark_sprompt_5e', help='Split-omnibenchmark s-prompt configs')
    elif config == 'omnibenchmark_dualprompt':
        from configs.omnibenchmark_dualprompt import get_args_parser
        config_parser = subparser.add_parser('omnibenchmark_dualprompt', help='Split-omnibenchmark dualprompt configs')



    else:
        raise NotImplementedError

    get_args_parser(config_parser)
    
    if debug_mode == 0:
        args = parser.parse_args()
    else:
        args = parser.parse_args(['{}'.format(config)]) # need to select the sub_parser

    args.config = config
    return args

def main(args, debug_mode=0):
    utils.init_distributed_mode(args)
    
    if debug_mode:
        args.output_dir = './debug' + args.output_dir[1:]
    else:
        pass
    
    args.output_dir = args.output_dir + '/' + args.config + '/test_{}'.format(random.randint(10000,90000))
    
    if args.output_dir and utils.is_main_process():
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
        # save agrs
        args_dict = vars(args)
        with open(args.output_dir + "/args.json", "w") as json_file:
            json.dump(args_dict, json_file, indent=2)

        # log
        log_out = args.output_dir + '/output.log'   
        sys.stdout = Logger(log_out)
        
    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    if hasattr(args, 'train_inference_task_only') and args.train_inference_task_only:
        import trainers.tii_trainer as tii_trainer
        tii_trainer.train(args)
    elif 'hideprompt' in args.config and not args.train_inference_task_only:
        import trainers.hideprompt_trainer as hideprompt_trainer
        hideprompt_trainer.train(args)
    elif 'l2p' in args.config or 'dualprompt' in args.config or 'sprompt' in args.config:
        #TODO: fix all possible implementations
        if 'sprompt' in args.config:
            args.config = 'sprompt_plus'
        import trainers.dp_trainer as dp_trainer
        dp_trainer.train(args)
    elif 'hidelora' in args.config and not args.train_inference_task_only:
        import trainers.hidelora_trainer as hidelora_trainer
        hidelora_trainer.train(args)
    elif 'continual_lora' in args.config:
        import trainers.continual_lora_trainer as continual_lora_trainer
        continual_lora_trainer.train(args)
    else:
        raise NotImplementedError

if __name__ == '__main__':
    
    DEBUG_MODE = 0
    # import os
    # os.environ['CUDA_VISIBLE_DEVICES'] = "2"
    
    args = get_args(debug_mode=DEBUG_MODE)
    print(args)
    main(args, debug_mode=DEBUG_MODE)
