import torch
import numpy as np
import time
from sklearn.decomposition import PCA
from auto.common_utils import update_memory_prefix
from auto.common_utils import row_cosine_similarity_sum, filter_memory_with_Spre, get_dict_topk_max
import math

def get_prefix_matrix(data_loader, model, device, fake_idx):
    model.eval()
    count = 1
    representation_g = {}
    representation_e = {}
    num_meets = False
    with torch.no_grad():
        for i in range(20):
            for input, target in data_loader:
                input = input.to(device, non_blocking=True)
                # prefix = torch.randn(1, 2, 5, 12, 64)
                

                _ = model(input, task_id=fake_idx, cls_features='only_sampling', train=True)
                del _

                # sprompt_plus, no g_prompt
                # for layer in model.g_prefix_feature:
                #     if layer not in representation_g:
                #         representation_g[layer] = {"key": []}
                #         # model.g_prefix_feature[layer]["key"]: [24, 12, 64]
                #     representation_g[layer]["key"].append(model.g_prefix_feature[layer]["key"])
                for layer in model.e_prefix_feature:
                    if layer not in representation_e:
                        representation_e[layer] = {"key": []}
                    representation_e[layer]["key"].append(model.e_prefix_feature[layer]["key"])
                count += 1

                if count > 768: 
                    num_meets = True
                    for layer in representation_e:
                        for item in representation_e[layer]:
                            representation_e[layer][item] = torch.cat(representation_e[layer][item])
                            representation_e[layer][item] = representation_e[layer][item].detach().cpu().numpy()
                            representation_e[layer][item] = representation_e[layer][item].reshape(representation_e[layer][item].shape[0], -1)
                            rep = representation_e[layer][item]
                            pca = PCA(n_components=50)
                            pca = pca.fit(rep)
                            rep = pca.transform(rep)
                            representation_e[layer][item] = rep

                    break
            
            if num_meets:
                break
            
    torch.cuda.empty_cache()

    return representation_g, representation_e

def grad_proj(task_id, args, model, feature_prefix_gt, feature_prefix_et, prompt_id, init_with=None, pretrained_subspace_g=None,
              local_forward_type=None):

    #1 Grad Projection to prevent pretrained knowledge from forgetting, a soft constraint
    threshold2 = args.threshold2
    if args.use_pre_gradient_constraint:
        for k, (m, params) in enumerate(model.named_parameters()):
            if m == "e_prompt.prompt":
                old_shape = params.grad.data[0][0][prompt_id].shape

                for i in range(5):
                    xx = params.grad.data[i][0][prompt_id]
                    xx_minus = threshold2 * torch.matmul(xx.view(args.length, 768), pretrained_subspace_g[i]['key']).view(old_shape)
                    params.grad.data[i][0][prompt_id] = xx - xx_minus
    else:
        pass
            
    #2 Grad Projection to prevent old knowledge from forgetting, a hard constraint
    if local_forward_type == 'add':
        pass
    elif local_forward_type =='update':
        feature_prefix_et = feature_prefix_et[init_with]

        for k, (m, params) in enumerate(model.named_parameters()):
            if m == "e_prompt.prompt":
                old_shape = params.grad.data[0][0][prompt_id].shape 

                for i in range(5):
                    yy = params.grad.data[i][0][prompt_id]
                    yy_minus = torch.matmul(yy.view(args.length, 768), feature_prefix_et[i]['key']).view(old_shape)
                    params.grad.data[i][0][prompt_id] = yy - yy_minus
    else:
        raise NotImplementedError


def get_angel(model, args, prompt_id, feature_prefix_gt, feature_prefix_et, pretrained_subspace_g):


    e_pre = []
    e_all = []
    for k, (m, params) in enumerate(model.named_parameters()):
        if m == "e_prompt.prompt":

            for i in range(5):
                grad_prompt = params.grad.data[i][0][prompt_id].view(args.length, 768).detach()
                grad_prompt_on_Spre = torch.matmul(grad_prompt, pretrained_subspace_g[i]['key'])
                grad_prompt_on_Sall = torch.matmul(grad_prompt, feature_prefix_et[i]['key'])

                e_pre.append(row_cosine_similarity_sum(grad_prompt, grad_prompt_on_Spre))
                e_all.append(row_cosine_similarity_sum(grad_prompt, grad_prompt_on_Sall))
    return np.mean(e_pre), np.mean(e_all)


def update_memory(args, data_loader, model, device, threshold, feature_prefix_g, feature_prefix_e, fake_idx, 
                  local_forward_type, pretrained_subspace):
    
    if args.no_auto:
        # baseline
        print('>>> : no need to build memory')
        
        return {}, {}, None, None
    else:
        print('>>> : need to build memory')

        time1 = time.time()
        prefix_rep_g, prefix_rep_e = get_prefix_matrix(data_loader, model, device, fake_idx) 
        time2 = time.time()
        print('get_prefix_matrix use:', time2 - time1)

        feature_prefix_g = update_memory_prefix(prefix_rep_g, threshold, feature_prefix_g)
        feature_prefix_e = update_memory_prefix(prefix_rep_e, threshold, feature_prefix_e)

        feature_prefix_gt = {0: {}, 1: {}}
        feature_prefix_et = {0: {}, 1: {}, 2: {}, 3: {}, 4: {}}

        for layer in feature_prefix_e:
            for item in feature_prefix_e[layer]:
                temp_feature = feature_prefix_e[layer][item].reshape(feature_prefix_e[layer][item].shape[0], -1)
                Uf = torch.Tensor(np.dot(temp_feature, temp_feature.transpose())).to(device)
                print('e', layer, item, Uf.size())
                feature_prefix_et[layer][item] = Uf
                print("item", item)
        
        return feature_prefix_g, feature_prefix_e, feature_prefix_gt, feature_prefix_et


def dec_with_memory(model, criterion, data_loader, optimizer, device, max_norm, 
                    task_id, args, task_wise_fpgt, task_wise_fpet, available_mini_model_list, class_mask,
                    pretrained_subspace_g):
    if args.no_auto == 1:
        # baseline:
        local_forward_type = 'add'
        init_with = None
        enhance_id = None
    else:
        print('args.config: ', args.config)
        local_forward_type, init_with, enhance_id = dec_with_memory_sprompt_plus(
            model=model, criterion=criterion, data_loader=data_loader, optimizer=optimizer, 
            device=device, max_norm=max_norm, task_id=task_id, args=args, task_wise_fpgt=task_wise_fpgt, 
            task_wise_fpet=task_wise_fpet, available_mini_model_list=available_mini_model_list, 
            class_mask=class_mask,
            pretrained_subspace_g=pretrained_subspace_g)   

    return local_forward_type, init_with, enhance_id

def dec_with_memory_sprompt_plus(model, criterion, data_loader, optimizer, device, max_norm, 
                    task_id, args, task_wise_fpgt, task_wise_fpet, available_mini_model_list, class_mask,
                    pretrained_subspace_g):
    """
        S-Prompts Learning with Pre-trained Transformers: An Occam's Razor for Domain Incremental Learning
    """

    all_mini_model_list = np.unique(list(available_mini_model_list.values()))
    all_mini_model_angle_list_pre = {i:None for i in all_mini_model_list}
    all_mini_model_angle_list_all = {i:None for i in all_mini_model_list}

    for sub_item in all_mini_model_list:
        model.train()
        
        E_PRE, E_ALL = [], []
        # count = 1
        for input, target in data_loader:
            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            
            output = model(input, task_id=sub_item, cls_features='only_sampling', train=True)          
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
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            
            # Grad Projection
            e_pre, e_all = get_angel(
                model=model, args=args, prompt_id=prompt_id, 
                feature_prefix_gt = task_wise_fpgt[sub_item], 
                feature_prefix_et = task_wise_fpet[sub_item],
                pretrained_subspace_g=pretrained_subspace_g,
                )

            E_PRE.append(e_pre)
            E_ALL.append(e_all)

            model.zero_grad()

        # math.degrees(math.acos(0.18))
        
        all_mini_model_angle_list_pre[sub_item] = np.stack(E_PRE)
        all_mini_model_angle_list_all[sub_item] = np.stack(E_ALL)

    mean_all_mini_model_angle_list_pre = {i:None for i in all_mini_model_list}
    mean_all_mini_model_angle_list_all = {i:None for i in all_mini_model_list}

    for key, value in all_mini_model_angle_list_pre.items():

        xx = np.mean(value)
        mean_all_mini_model_angle_list_pre[key] =  math.degrees(math.acos(xx))
        angle_epsilon = math.degrees(math.acos(xx))
    for key, value in all_mini_model_angle_list_all.items():

        yy = np.mean(value)
        mean_all_mini_model_angle_list_all[key] = math.degrees(math.acos(yy))

    print('>>> angle list pre:', mean_all_mini_model_angle_list_pre)
    print('>>> angle list all:', mean_all_mini_model_angle_list_all)

    print('>>> angle_epsilon: ', angle_epsilon)
    print('>>> all angle list:=================================================================')

    (max_key, max_value) = max(mean_all_mini_model_angle_list_all.items(), key=lambda x: x[1])
    # min value and key
    (min_key, min_value) = min(mean_all_mini_model_angle_list_all.items(), key=lambda x: x[1])
    print('>>> done!===========================================================================')
    print('>>> max_key:', max_key, '>>> max_value', max_value)
    print('>>> min_key:', min_key, '>>> min_value', min_value)
    print('>>> done!===========================================================================')


    if args.model_num == 0:

        if max_value > angle_epsilon:
            local_forward_type = 'update'
            init_with = int(max_key)
            # init_with = int(min_key)
        else:

            local_forward_type = 'add'
            init_with = None

    else:
        pass

    if args.use_old_subspace_forward:
        top_k_keys, top_k_values = get_dict_topk_max(mean_all_mini_model_angle_list_all, args.topk_old_subspace)
        print('>>> top_k_keys: ', top_k_keys)

        enhance_id = top_k_keys
    else:
        enhance_id = None



    return local_forward_type, init_with, enhance_id