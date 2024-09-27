# ------------------------------------------
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
# ------------------------------------------
# Modification:
# Added code for dualprompt implementation
# -- Jaeho Lee, dlwogh9344@khu.ac.kr
# ------------------------------------------
"""
Train and eval functions used in main.py
"""
import math
import sys
import os
import datetime
import json
from typing import Iterable
from pathlib import Path
from tqdm import * 
import torch
import torch.distributed as dist
import numpy as np
from torch import optim
from timm.utils import accuracy
from timm.optim import create_optimizer
from torch.distributions.multivariate_normal import MultivariateNormal
import utils
import copy
import pickle
from sklearn.decomposition import PCA
from auto.common_utils import pars_preparing, modify_available_list

def train_one_epoch(
    model: torch.nn.Module, 
    original_model: torch.nn.Module,
    criterion, data_loader: Iterable, 
    optimizer: torch.optim.Optimizer,
    device: torch.device, 
    epoch: int, 
    max_norm: float = 0,
    set_training_mode=True, 
    task_id=-1, 
    class_mask=None, 
    args=None, 
    local_forward_type=None, 
    task_wise_fpgt=None,
    task_wise_fpet=None,
    init_with=None,
    grad_proj=None,
    pretrained_subspace_g=None,
    enhance_id=None,
    ):
    
    model.train(set_training_mode)
    original_model.eval()

    if args.distributed and utils.get_world_size() > 1:
        data_loader.sampler.set_epoch(epoch)

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('Lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('Loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = f'Train: Epoch[{epoch + 1:{int(math.log10(args.epochs)) + 1}}/{args.epochs}]'

    for input, target in metric_logger.log_every(data_loader, args.print_freq, header):
        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        with torch.no_grad():
            if original_model is not None:
                output = original_model(input)
                cls_features = output['pre_logits']
            else:
                cls_features = None

        output = model(
            input, task_id=task_id, 
            cls_features=cls_features,
            train=set_training_mode, enhance_id=enhance_id)
        
        logits = output['logits']
        prompt_id = output['prompt_idx'][0][0]
        
        # here is the trick to mask out classes of non-current tasks
        if args.train_mask and class_mask is not None:
            mask = class_mask[task_id]
            not_mask = np.setdiff1d(np.arange(args.nb_classes), mask)
            not_mask = torch.tensor(not_mask, dtype=torch.int64).to(device)
            logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))

        loss = criterion(logits, target)  # base criterion (CrossEntropyLoss)
        if args.pull_constraint and 'reduce_sim' in output:
            loss = loss - args.pull_constraint_coeff * output['reduce_sim']

        acc1, acc5 = accuracy(logits, target, topk=(1, 5))

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        
        # Grad Projection
        if args.no_auto:
            pass
        else:
            grad_proj(task_id=task_id, args=args, model=model, feature_prefix_gt=task_wise_fpgt, 
                    feature_prefix_et=task_wise_fpet, prompt_id=prompt_id, init_with=init_with,
                    pretrained_subspace_g=pretrained_subspace_g, local_forward_type=local_forward_type)
        
        optimizer.step()

        torch.cuda.synchronize()
        metric_logger.update(Loss=loss.item())
        metric_logger.update(Lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters['Acc@1'].update(acc1.item(), n=input.shape[0])
        metric_logger.meters['Acc@5'].update(acc5.item(), n=input.shape[0])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    # print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(
    model: torch.nn.Module, 
    original_model: torch.nn.Module, 
    data_loader,
    device, 
    task_id=-1, 
    class_mask=None, 
    target_task_map=None, 
    args=None, 
    constant_target_task_map=None, 
    available_mini_model_list=None
    ):
    
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test: [Task {}]'.format(task_id + 1)

    # switch to evaluation mode
    model.eval()
    original_model.eval()

    with torch.no_grad():
        for input, target in metric_logger.log_every(data_loader, args.print_freq, header):
            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            # compute output

            if original_model is not None:
                output = original_model(input)
                cls_features = output['pre_logits']
            else:
                cls_features = None


            output = model(input, task_id=task_id, cls_features=cls_features, available_mini_model_list=available_mini_model_list)
            logits = output['logits']
            promtp_idx = output['prompt_idx']  # tensor B x topk

            #if args.separated_head:
            #    class_mask = torch.tensor(class_mask, dtype=torch.int64).to(device)
            #    mask = class_mask[promtp_idx.squeeze(-1)]
            #    logits_mask = torch.ones_like(logits, device=device) * float('-inf')
            #    logits_mask = logits_mask.scatter_(1, mask, 0.0)
            #    logits = logits + logits_mask

            if args.task_inc and class_mask is not None:
                # adding mask to output logits
                mask = class_mask[task_id]
                mask = torch.tensor(mask, dtype=torch.int64).to(device)
                logits_mask = torch.ones_like(logits, device=device) * float('-inf')
                logits_mask = logits_mask.index_fill(1, mask, 0.0)
                logits = logits + logits_mask

            loss = criterion(logits, target)

            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
            task_inference_acc = utils.task_inference_accuracy(promtp_idx, target, constant_target_task_map)

            metric_logger.meters['Loss'].update(loss.item())
            metric_logger.meters['Acc@1'].update(acc1.item(), n=input.shape[0])
            metric_logger.meters['Acc@5'].update(acc5.item(), n=input.shape[0])
            metric_logger.meters['Acc@task'].update(task_inference_acc.item(), n=input.shape[0])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print(
        '* Acc@task {task_acc.global_avg:.3f} Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
        .format(task_acc=metric_logger.meters['Acc@task'],
                top1=metric_logger.meters['Acc@1'], top5=metric_logger.meters['Acc@5'],
                losses=metric_logger.meters['Loss']))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_till_now(
    model: torch.nn.Module, 
    original_model: torch.nn.Module, 
    data_loader,
    device, 
    task_id=-1, 
    class_mask=None, 
    target_task_map=None,
    acc_matrix=None, 
    args=None, 
    constant_target_task_map=None,
    available_mini_model_list=None, 
    ):
    
    stat_matrix = np.zeros((4, args.num_tasks))  # 3 for Acc@1, Acc@5, Loss

    for i in range(task_id + 1):
        test_stats = evaluate(
            model=model, original_model=original_model, data_loader=data_loader[i]['val'],
            device=device, task_id=i, class_mask=class_mask, target_task_map=target_task_map,
            args=args, constant_target_task_map=constant_target_task_map, 
            available_mini_model_list=available_mini_model_list
            )

        stat_matrix[0, i] = test_stats['Acc@1']
        stat_matrix[1, i] = test_stats['Acc@5']
        stat_matrix[2, i] = test_stats['Loss']
        stat_matrix[3, i] = test_stats['Acc@task']

        acc_matrix[i, task_id] = test_stats['Acc@1']

    avg_stat = np.divide(np.sum(stat_matrix, axis=1), task_id + 1)

    diagonal = np.diag(acc_matrix)

    result_str = "[Average accuracy till task{}]\tAcc@task: {:.4f}\tAcc@1: {:.4f}\tAcc@5: {:.4f}\tLoss: {:.4f}".format(
        task_id + 1,
        avg_stat[3],
        avg_stat[0],
        avg_stat[1],
        avg_stat[2])
    if task_id > 0:
        forgetting = np.mean((np.max(acc_matrix, axis=1) -
                              acc_matrix[:, task_id])[:task_id])
        backward = np.mean((acc_matrix[:, task_id] - diagonal)[:task_id])

        result_str += "\tForgetting: {:.4f}\tBackward: {:.4f}".format(forgetting, backward)
    print(result_str)

    if utils.is_main_process():
        # to save the stat_matrix
        file_name = args.output_dir + '/array_log.txt'
        # Path(file_name).mkdir(parents=True, exist_ok=True)
        
        with open(file_name, 'a') as file:
            file.write('\n\n')
            file.write('-----current_task_{}-----\n'.format(task_id + 1))

            for row in stat_matrix:
                formatted_row = ' '.join('{:8.4f}'.format(num) for num in row)
                file.write(formatted_row + '\n')

        print(f"NumPy array :stat_matrix saved to {file_name}")
        
    return test_stats


def train_and_evaluate(
    model: torch.nn.Module, model_without_ddp: torch.nn.Module, 
    original_model: torch.nn.Module,
    criterion, data_loader: Iterable, 
    data_loader_per_cls: Iterable, optimizer: torch.optim.Optimizer, 
    lr_scheduler, device: torch.device, 
    class_mask=None, target_task_map=None, args=None, 
    ):
    
    """
        auto_module, pkl import 
    """
    if 'l2p' in args.config:
        print('args.config: ', args.config)
        from auto.l2p.memory import dec_with_memory, grad_proj, update_memory
    elif 'dualprompt' in args.config:
        print('args.config: ', args.config)
        from auto.dualprompt.memory import dec_with_memory, grad_proj, update_memory
    elif 'sprompt_plus' in args.config:
        print('args.config: ', args.config)
        from auto.sprompt_plus.memory import dec_with_memory, grad_proj, update_memory
    elif 'sprompt_vallina' in args.config:
        print('args.config: ', args.config)
        from auto.sprompt_vallina.memory import dec_with_memory, grad_proj, update_memory
    elif 'hideprompt' in args.config:
        print('args.config: ', args.config)
        from auto.hidep.memory import dec_with_memory, grad_proj, update_memory
    else:
        raise NotImplementedError 

    acc_matrix = np.zeros((args.num_tasks, args.num_tasks))

    task_wise_fpg = {}
    task_wise_fpe = {}
    task_wise_fpgt = {}
    task_wise_fpet = {}
    
    feature_prefix_g = {} 
    feature_prefix_e = {} 
    feature_prefix_gt = None 
    feature_prefix_et = None 
    
    available_mini_model_list = {}  
    constant_target_task_map = copy.deepcopy(target_task_map) 

    pretrained_subspace, pretrained_subspace_g = pars_preparing(args, device, data_loader, original_model, class_num=len(target_task_map.keys()))

    for task_id in range(args.num_tasks):
        print('>>> processing on task: {}'.format(task_id))
        #TODO
        if task_id == 0:
            local_forward_type = 'add'
            init_with = None
            enhance_id = None
        else:
            local_forward_type, init_with, enhance_id = dec_with_memory(
                model=model, criterion=criterion, data_loader=data_loader[task_id]['train'], 
                optimizer=optimizer, device=device, max_norm=args.clip_grad, task_id=task_id, 
                args=args, task_wise_fpgt=task_wise_fpgt, task_wise_fpet=task_wise_fpet, 
                available_mini_model_list=available_mini_model_list,
                class_mask=class_mask, 
                pretrained_subspace_g=pretrained_subspace_g,
                )
    
        available_mini_model_list = modify_available_list(
            task_id=task_id, available_mini_model_list=available_mini_model_list, 
            local_forward_type=local_forward_type, init_with=init_with, 
            )
        
        # Transfer previous learned prompt params to the new prompt
        # if args.prompt_pool and args.shared_prompt_pool:
        if task_id == 0 and local_forward_type == 'add' and init_with is None:
            print('>>> : task0, no need to re-init params')           
            pass
        else:
            assert task_id > 0, 'xxx'
            if local_forward_type == 'add' and init_with is None:

                print('>>> : task{}, no need to re-init params'.format(task_id))    
                pass
            elif local_forward_type == 'update' and init_with is not None:

                print('>>> : task{}, re-init params with task{}'.format(task_id, init_with))    

                prev_start = init_with * args.top_k
                prev_end = (init_with + 1) * args.top_k

                cur_start = task_id * args.top_k
                cur_end = (task_id + 1) * args.top_k

                if (prev_end > args.size) or (cur_end > args.size):
                    pass
                else:
                    cur_idx = (
                        slice(None), slice(None), slice(cur_start, cur_end)) if args.use_prefix_tune_for_e_prompt else (
                        slice(None), slice(cur_start, cur_end))
                    prev_idx = (
                        slice(None), slice(None),
                        slice(prev_start, prev_end)) if args.use_prefix_tune_for_e_prompt else (
                        slice(None), slice(prev_start, prev_end))

                    with torch.no_grad():
                        if args.distributed:
                            if model.module.e_prompt.prompt.grad is not None:
                                # print('model.module.e_prompt.prompt.grad  =\n', model.module.e_prompt.prompt.grad)
                                print('e_prompt.prompt.grad is not None')
                                model.module.e_prompt.prompt.grad.zero_()
                            else:
                                print('model.module.e_prompt.prompt.requires_grad  =\n', model.module.e_prompt.prompt.requires_grad)
                                model.module.e_prompt.prompt.requires_grad = True
                            model.module.e_prompt.prompt[cur_idx] = model.module.e_prompt.prompt[prev_idx]
                            optimizer.param_groups[0]['params'] = model.module.parameters()
                        else:
                            if model.e_prompt.prompt.grad is not None:
                                # print('model.e_prompt.prompt.grad  =\n', model.e_prompt.prompt.grad)
                                print('e_prompt.prompt.grad is not None')
                                model.e_prompt.prompt.grad.zero_()
                            else:
                                print('model.e_prompt.prompt.requires_grad  =\n', model.e_prompt.prompt.requires_grad)
                                model.e_prompt.prompt.requires_grad = True
                            model.e_prompt.prompt[cur_idx] = model.e_prompt.prompt[prev_idx]
                            optimizer.param_groups[0]['params'] = model.parameters()


        # Transfer previous learned prompt param keys to the new prompt
        # if args.prompt_pool and args.shared_prompt_key:
        if task_id == 0 and local_forward_type == 'add' and init_with is None:
            print('>>> : task0, no need to re-init keys')   
            pass
        else:
            assert task_id > 0, 'yyy'
            if local_forward_type == 'add' and init_with is None:
                print('>>> : task{}, no need to re-init keys'.format(task_id))    
                pass
            elif local_forward_type == 'update' and init_with is not None:
                print('>>> : task{}, re-init keys with task{}'.format(task_id, init_with))    
            
                prev_start = init_with * args.top_k
                prev_end = (init_with + 1) * args.top_k

                cur_start = task_id * args.top_k
                cur_end = (task_id + 1) * args.top_k
                

                if (prev_end > args.size) or (cur_end > args.size):
                    pass
                else:
                    cur_idx = (
                        slice(cur_start, cur_end)) if args.use_prefix_tune_for_e_prompt else (
                        slice(None), slice(cur_start, cur_end))
                    prev_idx = (
                        slice(prev_start, prev_end)) if args.use_prefix_tune_for_e_prompt else (
                        slice(None), slice(prev_start, prev_end))

                    with torch.no_grad():
                        if args.distributed:
                            if model.module.e_prompt.prompt_key.grad is not None:
                                # print('model.module.e_prompt.prompt_key.grad  =\n', model.module.e_prompt.prompt_key.grad)
                                print('e_prompt.prompt.grad is not None')
                                model.module.e_prompt.prompt_key.grad.zero_()
                            else:
                                print('model.module.e_prompt.prompt_key.requires_grad  =\n', model.module.e_prompt.prompt_key.requires_grad)
                                model.module.e_prompt.prompt_key.requires_grad = True
                            model.module.e_prompt.prompt_key[cur_idx] = model.module.e_prompt.prompt_key[prev_idx]
                            # optimizer.param_groups[0]['params'] = model.module.parameters()
                        else:
                            if model.e_prompt.prompt_key.grad is not None:
                                # print('model.e_prompt.prompt_key.grad  =\n', model.e_prompt.prompt_key.grad)
                                print('e_prompt.prompt.grad is not None')
                                model.e_prompt.prompt_key.grad.zero_()
                            else:
                                print('model.e_prompt.prompt_key.requires_grad  =\n', model.e_prompt.prompt_key.requires_grad)
                                model.e_prompt.prompt_key.requires_grad = True
                            model.e_prompt.prompt_key[cur_idx] = model.e_prompt.prompt_key[prev_idx]
                            # optimizer.param_groups[0]['params'] = model.parameters()

        #5 Create new optimizer for each task to clear optimizer status
        if task_id > 0 and args.reinit_optimizer:
            optimizer = create_optimizer(args, model)

        #6 training
        for epoch in tqdm(range(args.epochs)):
            train_stats = train_one_epoch(
                model=model, original_model=original_model, criterion=criterion,
                data_loader=data_loader[task_id]['train'], optimizer=optimizer,
                device=device, epoch=epoch, max_norm=args.clip_grad,
                set_training_mode=True, task_id=task_id, class_mask=class_mask, args=args, 
                task_wise_fpgt=task_wise_fpgt,
                task_wise_fpet=task_wise_fpet,
                init_with=init_with,
                local_forward_type=local_forward_type,
                grad_proj=grad_proj,
                pretrained_subspace_g=pretrained_subspace_g,
                enhance_id=enhance_id
                )

            if lr_scheduler:
                lr_scheduler.step(epoch)

        #7 update memory and modify the global proj
        
        if init_with is None and local_forward_type == 'add':

            sub_feature_prefix_g = {} 
            sub_feature_prefix_e = {} 
            sub_feature_prefix_gt = None 
            sub_feature_prefix_et = None 
        elif init_with is not None and local_forward_type == 'update':

            sub_feature_prefix_g = task_wise_fpg[init_with] 
            sub_feature_prefix_e = task_wise_fpe[init_with]
            sub_feature_prefix_gt = task_wise_fpgt[init_with] 
            sub_feature_prefix_et = task_wise_fpet[init_with] 
            

            del task_wise_fpg[init_with]
            del task_wise_fpe[init_with]
            del task_wise_fpgt[init_with] 
            del task_wise_fpet[init_with] 
            
        else:
            pass
        
        if args.no_auto == 1:
            # baseline
            pass
        else:
            sub_feature_prefix_g, sub_feature_prefix_e, sub_feature_prefix_gt, sub_feature_prefix_et = update_memory(
                args=args, data_loader=data_loader[task_id]['mem'], 
                model=model, device=device, threshold=args.threshold, 
                feature_prefix_g=sub_feature_prefix_g, feature_prefix_e=sub_feature_prefix_e,
                fake_idx=task_id, local_forward_type=local_forward_type,
                pretrained_subspace=pretrained_subspace
                ) #  threshold=0.50       
        
        task_wise_fpg[task_id] = sub_feature_prefix_g
        task_wise_fpe[task_id] = sub_feature_prefix_e
        task_wise_fpgt[task_id] = sub_feature_prefix_gt
        task_wise_fpet[task_id] = sub_feature_prefix_et

        if init_with is None and local_forward_type == 'add':
            pass
        elif init_with is not None and local_forward_type == 'update':
            for key, value in constant_target_task_map.items():
                if value == init_with:
                    constant_target_task_map[key] = task_id
        else:
            pass
        # ==================================================================================================

        test_stats = evaluate_till_now(
            model=model, original_model=original_model, data_loader=data_loader,
            device=device, task_id=task_id, class_mask=class_mask, target_task_map=target_task_map,
            acc_matrix=acc_matrix, args=args, constant_target_task_map=constant_target_task_map,
            available_mini_model_list=available_mini_model_list, 
            )

        if args.output_dir and utils.is_main_process(): # and task_id == args.num_tasks - 1: 
        # if False: # for store-efficiency, no ckpt will be saved
            Path(os.path.join(args.output_dir, 'checkpoint')).mkdir(parents=True, exist_ok=True)

            checkpoint_path = os.path.join(args.output_dir, 'checkpoint/task{}_checkpoint.pth'.format(task_id + 1))
            
            # for utility, only save the prompt params and fc params
            all_model_state_dict = model_without_ddp.state_dict()
            for key in list(all_model_state_dict.keys()):
                if key.startswith(tuple(args.freeze)):
                    # print(key)
                    all_model_state_dict.pop(key, None)

            state_dict = {
                'model': all_model_state_dict,
                'optimizer': optimizer.state_dict(),
                'args': args,
            }
            if args.sched is not None and args.sched != 'constant':
                state_dict['lr_scheduler'] = lr_scheduler.state_dict()

            utils.save_on_master(state_dict, checkpoint_path)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     }

        if args.output_dir and utils.is_main_process():
            with open(os.path.join(args.output_dir,
                                   '{}_stats.txt'.format(datetime.datetime.now().strftime('log_%Y_%m_%d_%H_%M'))),
                      'a') as f:
                f.write(json.dumps(log_stats) + '\n')