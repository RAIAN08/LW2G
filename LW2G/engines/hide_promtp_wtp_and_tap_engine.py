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

import torch
import torch.distributed as dist
import numpy as np
from tqdm import *

from timm.utils import accuracy
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from torch import optim
import utils
from torch.distributions.multivariate_normal import MultivariateNormal
from utils import is_dist_avail_and_initialized 
import copy
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
        target_task_map=None, 
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
                logits = output['logits']

                if args.train_mask and class_mask is not None:
                    mask = []
                    for id in range(task_id + 1):
                        mask.extend(class_mask[id])
                    not_mask = np.setdiff1d(np.arange(args.nb_classes), mask)
                    not_mask = torch.tensor(not_mask, dtype=torch.int64).to(device)
                    logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))
                    prompt_id = torch.max(logits, dim=1)[1]
                    # translate cls to task_id
                    prompt_id = torch.tensor([target_task_map[v.item()] for v in prompt_id], device=device).unsqueeze(
                        -1)
                else:
                    prompt_id = None
            else:
                raise NotImplementedError("original model is None")
            
        output = model(
            input, task_id=task_id, prompt_id=prompt_id, train=set_training_mode,
            prompt_momentum=args.prompt_momentum, enhance_id=enhance_id,
            cls_features=None,
            )
        
        logits = output['logits']
        prompt_id = output['prompt_idx'][0][0]

        # here is the trick to mask out classes of non-current tasks
        if args.train_mask and class_mask is not None:
            mask = class_mask[task_id]
            not_mask = np.setdiff1d(np.arange(args.nb_classes), mask)
            not_mask = torch.tensor(not_mask, dtype=torch.int64).to(device)
            logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))

        loss = criterion(logits, target)  # base criterion (CrossEntropyLoss)
        # TODO add contrastive loss
        loss += orth_loss(output['pre_logits'], target, device, args)
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
    i=-1, 
    task_id=-1, 
    class_mask=None, 
    target_task_map=None, 
    args=None, 
    constant_target_task_map=None, 
    available_mini_model_list=None
    ):

    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test: [Task {}]'.format(i + 1)

    # switch to evaluation mode
    model.eval()
    original_model.eval()

    with torch.no_grad():
        for input, target in metric_logger.log_every(data_loader, args.print_freq, header):
            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            # compute output
            with torch.no_grad():
                if original_model is not None:
                    output = original_model(input)
                    logits = output['logits']
                    if args.train_mask and class_mask is not None:
                        mask = []
                        for id in range(task_id + 1):
                            mask.extend(class_mask[id])
                        not_mask = np.setdiff1d(np.arange(args.nb_classes), mask)
                        not_mask = torch.tensor(not_mask, dtype=torch.int64).to(device)
                        logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))
                    prompt_id = torch.max(logits, dim=1)[1]
                    # translate cls to task_id
                    prompt_id = torch.tensor([target_task_map[v.item()] for v in prompt_id], device=device).unsqueeze(
                        -1)
                    # print(prompt_id)
                else:
                    raise NotImplementedError("original model is None")

            output = model(input, task_id=task_id, prompt_id=prompt_id, available_mini_model_list=available_mini_model_list)
            logits = output['logits']
            promtp_idx = output['prompt_idx']  # tensor B x topk

            if args.task_inc and class_mask is not None:
                # adding mask to output logits
                mask = class_mask[i]
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
            device=device, i=i, task_id=task_id, class_mask=class_mask, target_task_map=target_task_map,
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


def train_and_evaluate(model: torch.nn.Module, model_without_ddp: torch.nn.Module, original_model: torch.nn.Module,
                       criterion, data_loader: Iterable, data_loader_per_cls: Iterable,
                       optimizer: torch.optim.Optimizer,
                       lr_scheduler,
                       device: torch.device,
                       class_mask=None, target_task_map=None, args=None, ):
    
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
    
    feature_prefix_g = {} #
    feature_prefix_e = {} 
    feature_prefix_gt = None 
    feature_prefix_et = None 

    available_mini_model_list = {} 
    constant_target_task_map = copy.deepcopy(target_task_map) 

    pretrained_subspace, pretrained_subspace_g = pars_preparing(args, device, data_loader, original_model, class_num=len(target_task_map.keys()))

    global cls_mean
    global cls_cov
    cls_mean = dict()
    cls_cov = dict()

    for task_id in range(args.num_tasks):
        print('>>> processing on task: {}'.format(task_id))
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
        
        if task_id > 0 and args.reinit_optimizer:
            if args.larger_prompt_lr:
                base_params = [p for name, p in model_without_ddp.named_parameters() if
                            'prompt' in name and p.requires_grad == True]
                base_fc_params = [p for name, p in model_without_ddp.named_parameters() if
                                'prompt' not in name and p.requires_grad == True]
                base_params = {'params': base_params, 'lr': args.lr, 'weight_decay': args.weight_decay}
                base_fc_params = {'params': base_fc_params, 'lr': args.lr * 0.1, 'weight_decay': args.weight_decay}
                network_params = [base_params, base_fc_params]
                optimizer = create_optimizer(args, network_params)
            else:
                optimizer = create_optimizer(args, model)
            
            if args.sched != 'constant':
                lr_scheduler, _ = create_scheduler(args, optimizer)
            elif args.sched == 'constant':
                lr_scheduler = None

        # load original model checkpoint
        if args.trained_original_model:
            original_checkpoint_path = os.path.join(args.trained_original_model,
                                                    'checkpoint/task{}_checkpoint.pth'.format(task_id + 1))
            if os.path.exists(original_checkpoint_path):
                print('Loading checkpoint from:', original_checkpoint_path)
                original_checkpoint = torch.load(original_checkpoint_path, map_location=device)
                original_model.load_state_dict(original_checkpoint['model'])
            else:
                print('No checkpoint found at:', original_checkpoint_path)
                return
        # if model already trained
        checkpoint_path = os.path.join(args.output_dir, 'checkpoint/task{}_checkpoint.pth'.format(task_id + 1))
        
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

        # # Transfer previous learned prompt param keys to the new prompt
        # # if args.prompt_pool and args.shared_prompt_key:
        # if task_id == 0 and local_forward_type == 'add' and init_with is None:
        #     print('>>> : task0, no need to re-init keys')   
        #     pass
        # else:
        #     assert task_id > 0, 'yyy'
        #     if local_forward_type == 'add' and init_with is None:
        #         print('>>> : task{}, no need to re-init keys'.format(task_id))    
        #         pass
        #     elif local_forward_type == 'update' and init_with is not None:
        #         print('>>> : task{}, re-init keys with task{}'.format(task_id, init_with))    
            
        #         prev_start = init_with * args.top_k
        #         prev_end = (init_with + 1) * args.top_k

        #         cur_start = task_id * args.top_k
        #         cur_end = (task_id + 1) * args.top_k
                

        #         if (prev_end > args.size) or (cur_end > args.size):
        #             pass
        #         else:
        #             cur_idx = (
        #                 slice(cur_start, cur_end)) if args.use_prefix_tune_for_e_prompt else (
        #                 slice(None), slice(cur_start, cur_end))
        #             prev_idx = (
        #                 slice(prev_start, prev_end)) if args.use_prefix_tune_for_e_prompt else (
        #                 slice(None), slice(prev_start, prev_end))

        #             with torch.no_grad():
        #                 if args.distributed:
        #                     if model.module.e_prompt.prompt_key.grad is not None:
        #                         # print('model.module.e_prompt.prompt_key.grad  =\n', model.module.e_prompt.prompt_key.grad)
        #                         print('e_prompt.prompt.grad is not None')
        #                         model.module.e_prompt.prompt_key.grad.zero_()
        #                     else:
        #                         print('model.module.e_prompt.prompt_key.requires_grad  =\n', model.module.e_prompt.prompt_key.requires_grad)
        #                         model.module.e_prompt.prompt_key.requires_grad = True
        #                     model.module.e_prompt.prompt_key[cur_idx] = model.module.e_prompt.prompt_key[prev_idx]
        #                     # optimizer.param_groups[0]['params'] = model.module.parameters()
        #                 else:
        #                     if model.e_prompt.prompt_key.grad is not None:
        #                         # print('model.e_prompt.prompt_key.grad  =\n', model.e_prompt.prompt_key.grad)
        #                         print('e_prompt.prompt.grad is not None')
        #                         model.e_prompt.prompt_key.grad.zero_()
        #                     else:
        #                         print('model.e_prompt.prompt_key.requires_grad  =\n', model.e_prompt.prompt_key.requires_grad)
        #                         model.e_prompt.prompt_key.requires_grad = True
        #                     model.e_prompt.prompt_key[cur_idx] = model.e_prompt.prompt_key[prev_idx]
        #                     # optimizer.param_groups[0]['params'] = model.parameters()

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
                enhance_id=enhance_id,
                target_task_map=target_task_map
                )

            if lr_scheduler:
                lr_scheduler.step(epoch)

        if args.prompt_momentum > 0 and task_id > 0:
            if args.use_prefix_tune_for_e_prompt:
                if args.distributed:
                    with torch.no_grad():
                        print(model.module.e_prompt.prompt[:, :, task_id].shape)
                        print(
                            model.module.e_prompt.prompt[:, :, 0:task_id].detach().clone().mean(dim=2, keepdim=True).shape)
                        model.module.e_prompt.prompt[:, :, task_id].copy_(
                            (1 - args.prompt_momentum) * model.module.e_prompt.prompt[:, :, task_id].detach().clone()
                            + args.prompt_momentum * model.module.e_prompt.prompt[:, :, 0:task_id].detach().clone().mean(
                                dim=2))
                else:
                    with torch.no_grad():
                        print(model.e_prompt.prompt[:, :, task_id].shape)
                        print(
                            model.e_prompt.prompt[:, :, 0:task_id].detach().clone().mean(dim=2, keepdim=True).shape)
                        model.e_prompt.prompt[:, :, task_id].copy_(
                            (1 - args.prompt_momentum) * model.e_prompt.prompt[:, :, task_id].detach().clone()
                            + args.prompt_momentum * model.e_prompt.prompt[:, :, 0:task_id].detach().clone().mean(
                                dim=2))
                        
        # compute mean and variance
        _compute_mean(model=model, data_loader=data_loader_per_cls, device=device, task_id=task_id,
                      class_mask=class_mask[task_id], args=args)

        if task_id > 0 and not args.not_train_ca:
            train_task_adaptive_prediction(model, args, device, class_mask, task_id)
        
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



        # if args.output_dir and utils.is_main_process():
        #     Path(os.path.join(args.output_dir, 'checkpoint')).mkdir(parents=True, exist_ok=True)

        #     checkpoint_path = os.path.join(args.output_dir, 'checkpoint/task{}_checkpoint.pth'.format(task_id + 1))
        #     state_dict = {
        #         'model': model_without_ddp.state_dict(),
        #         'optimizer': optimizer.state_dict(),
        #         'args': args,
        #     }
        #     if args.sched is not None and args.sched != 'constant':
        #         state_dict['lr_scheduler'] = lr_scheduler.state_dict()

        #     utils.save_on_master(state_dict, checkpoint_path)

        # log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
        #              **{f'test_{k}': v for k, v in test_stats.items()},
        #              }

        # if args.output_dir and utils.is_main_process():
        #     with open(os.path.join(args.output_dir,
        #                            '{}_stats.txt'.format(datetime.datetime.now().strftime('log_%Y_%m_%d_%H_%M'))),
        #               'a') as f:
        #         f.write(json.dumps(log_stats) + '\n')


@torch.no_grad()
def _compute_mean(model: torch.nn.Module, data_loader: Iterable, device: torch.device, task_id, class_mask=None,
                  args=None, ):
    model.eval()

    for cls_id in class_mask:
        data_loader_cls = data_loader[cls_id]['train']
        features_per_cls = []
        for i, (inputs, targets) in enumerate(data_loader_cls):
            inputs = inputs.to(device, non_blocking=True)
            features = model(inputs, task_id=task_id, train=True)['pre_logits']
            features_per_cls.append(features)
        features_per_cls = torch.cat(features_per_cls, dim=0)
        features_per_cls_list = [torch.zeros_like(features_per_cls, device=device) for _ in range(args.world_size)]

        if is_dist_avail_and_initialized():
            features_per_cls_list = [torch.zeros_like(features_per_cls, device=device) for _ in range(args.world_size)]
            dist.barrier()
            dist.all_gather(features_per_cls_list, features_per_cls)
        else:
            features_per_cls_list = [features_per_cls]
            
        if args.ca_storage_efficient_method == 'covariance':
            features_per_cls = torch.cat(features_per_cls_list, dim=0)
            # print(features_per_cls.shape)
            cls_mean[cls_id] = features_per_cls.mean(dim=0)
            cls_cov[cls_id] = torch.cov(features_per_cls.T) + (torch.eye(cls_mean[cls_id].shape[-1]) * 1e-4).to(device)
        
        if args.ca_storage_efficient_method == 'variance':
            features_per_cls = torch.cat(features_per_cls_list, dim=0)
            # print(features_per_cls.shape)
            cls_mean[cls_id] = features_per_cls.mean(dim=0)
            cls_cov[cls_id] = torch.diag(torch.cov(features_per_cls.T) + (torch.eye(cls_mean[cls_id].shape[-1]) * 1e-4).to(device))
        if args.ca_storage_efficient_method == 'multi-centroid':
            from sklearn.cluster import KMeans
            n_clusters = args.n_centroids
            features_per_cls = torch.cat(features_per_cls_list, dim=0).cpu().numpy()
            kmeans = KMeans(n_clusters=n_clusters)
            kmeans.fit(features_per_cls)
            cluster_lables = kmeans.labels_
            cluster_means = []
            cluster_vars = []
            for i in range(n_clusters):
               cluster_data = features_per_cls[cluster_lables == i]
               cluster_mean = torch.tensor(np.mean(cluster_data, axis=0), dtype=torch.float64).to(device)
               cluster_var = torch.tensor(np.var(cluster_data, axis=0), dtype=torch.float64).to(device)
               cluster_means.append(cluster_mean)
               cluster_vars.append(cluster_var)
            
            cls_mean[cls_id] = cluster_means
            cls_cov[cls_id] = cluster_vars


def train_task_adaptive_prediction(model: torch.nn.Module, args, device, class_mask=None, task_id=-1):
    model.train()
    run_epochs = args.crct_epochs
    crct_num = 0
    param_list = [p for n, p in model.named_parameters() if p.requires_grad and 'prompt' not in n]
    network_params = [{'params': param_list, 'lr': args.ca_lr, 'weight_decay': args.weight_decay}]
    if 'mae' in args.model or 'beit' in args.model:
        optimizer = optim.AdamW(network_params, lr=args.ca_lr / 10, weight_decay=args.weight_decay)
    else:
        optimizer = optim.SGD(network_params, lr=args.ca_lr, momentum=0.9, weight_decay=5e-4)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=run_epochs)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    for i in range(task_id):
        crct_num += len(class_mask[i])

    # TODO: efficiency may be improved by encapsulating sampled data into Datasets class and using distributed sampler.
    for epoch in range(run_epochs):

        sampled_data = []
        sampled_label = []
        num_sampled_pcls = args.batch_size * 5

        metric_logger = utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter('Lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        metric_logger.add_meter('Loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))

        if args.ca_storage_efficient_method in ['covariance', 'variance']:
            for i in range(task_id + 1):
                for c_id in class_mask[i]:
                    mean = torch.tensor(cls_mean[c_id], dtype=torch.float64).to(device)
                    cov = cls_cov[c_id].to(device)
                    if args.ca_storage_efficient_method == 'variance':
                        cov = torch.diag(cov)
                    m = MultivariateNormal(mean.float(), cov.float())
                    sampled_data_single = m.sample(sample_shape=(num_sampled_pcls,))
                    sampled_data.append(sampled_data_single)

                    sampled_label.extend([c_id] * num_sampled_pcls)

        elif args.ca_storage_efficient_method == 'multi-centroid':
            for i in range(task_id + 1):
               for c_id in class_mask[i]:
                   for cluster in range(len(cls_mean[c_id])):
                       mean = cls_mean[c_id][cluster]
                       var = cls_cov[c_id][cluster]
                       if var.mean() == 0:
                           continue
                       m = MultivariateNormal(mean.float(), (torch.diag(var) + 1e-4 * torch.eye(mean.shape[0]).to(mean.device)).float())
                       sampled_data_single = m.sample(sample_shape=(num_sampled_pcls,))
                       sampled_data.append(sampled_data_single)
                       sampled_label.extend([c_id] * num_sampled_pcls)
        else:
            raise NotImplementedError


        sampled_data = torch.cat(sampled_data, dim=0).float().to(device)
        sampled_label = torch.tensor(sampled_label).long().to(device)
        # print(sampled_data.shape)

        inputs = sampled_data
        targets = sampled_label

        sf_indexes = torch.randperm(inputs.size(0))
        inputs = inputs[sf_indexes]
        targets = targets[sf_indexes]
        # print(targets)

        for _iter in range(crct_num):
            inp = inputs[_iter * num_sampled_pcls:(_iter + 1) * num_sampled_pcls]
            tgt = targets[_iter * num_sampled_pcls:(_iter + 1) * num_sampled_pcls]
            outputs = model(inp, fc_only=True)
            logits = outputs['logits']

            if args.train_mask and class_mask is not None:
                mask = []
                for id in range(task_id + 1):
                    mask.extend(class_mask[id])
                # print(mask)
                not_mask = np.setdiff1d(np.arange(args.nb_classes), mask)
                not_mask = torch.tensor(not_mask, dtype=torch.int64).to(device)
                logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))

            loss = criterion(logits, tgt)  # base criterion (CrossEntropyLoss)
            acc1, acc5 = accuracy(logits, tgt, topk=(1, 5))

            if not math.isfinite(loss.item()):
                print("Loss is {}, stopping training".format(loss.item()))
                sys.exit(1)

            optimizer.zero_grad()
            loss.backward()
            #for name, p in model.named_parameters():
            #    if p.requires_grad and p.grad is None:
            #        print(name)
            optimizer.step()
            torch.cuda.synchronize()

            metric_logger.update(Loss=loss.item())
            metric_logger.update(Lr=optimizer.param_groups[0]["lr"])
            metric_logger.meters['Acc@1'].update(acc1.item(), n=inp.shape[0])
            metric_logger.meters['Acc@5'].update(acc5.item(), n=inp.shape[0])

            # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        # print("Averaged stats:", metric_logger)
        scheduler.step()


def orth_loss(features, targets, device, args):
    if cls_mean:
        # orth loss of this batch
        sample_mean = []
        for k, v in cls_mean.items():
            if isinstance(v, list):
                sample_mean.extend(v)
            else:
                sample_mean.append(v)
        sample_mean = torch.stack(sample_mean, dim=0).to(device, non_blocking=True)
        M = torch.cat([sample_mean, features], dim=0)
        sim = torch.matmul(M, M.t()) / 0.8
        loss = torch.nn.functional.cross_entropy(sim, torch.range(0, sim.shape[0] - 1).long().to(device))
        # print(loss)
        return args.reg * loss
    else:
        sim = torch.matmul(features, features.t()) / 0.8
        loss = torch.nn.functional.cross_entropy(sim, torch.range(0, sim.shape[0] - 1).long().to(device))
        return args.reg * loss
        # return 0.

