import torch
import numpy as np
from sklearn.decomposition import PCA
import heapq
import os
import pickle

def filter_memory_with_Spre(represent, pretrained_subspace):
    if len(represent) == 0:
        return {}
    
    # bulid a new represent dict
    new_represent = {} 
    for key, value in represent.items():
        new_represent[key] = {'key' : 0}

    for layer in represent:
        for item in represent[layer]:
            representation = represent[layer][item]
            # representation = np.matmul(representation, representation.T)
            feature = pretrained_subspace[layer][item]

            # Projected Representation onto pretrained_subspace
            act_hat = representation - np.dot(np.dot(feature, feature.transpose()), representation)
            new_represent[layer][item] = act_hat
    
    return new_represent


def update_memory_prefix(represent, threshold, features=None):
    for layer in represent:
        for item in represent[layer]:
            representation = represent[layer][item]
            representation = np.matmul(representation, representation.T)
            try:
                feature = features[layer][item]
            except:
                feature = None
            if feature is None:
                if layer not in features:
                    features[layer] = {}
                U, S, Vh = np.linalg.svd(representation, full_matrices=False)
                sval_total = (S ** 2).sum()
                sval_ratio = (S ** 2) / sval_total
                r = np.sum(np.cumsum(sval_ratio) < threshold)
                print('layer', layer, 'item', item, 'r', r)
                feature = U[:, 0:r]
            else:
                U1, S1, Vh1 = np.linalg.svd(representation, full_matrices=False)
                sval_total = (S1 ** 2).sum()
                # Projected Representation
                act_hat = representation - np.dot(np.dot(feature, feature.transpose()), representation)
                U, S, Vh = np.linalg.svd(act_hat, full_matrices=False)
                # criteria
                sval_hat = (S ** 2).sum()
                sval_ratio = (S ** 2) / sval_total
                accumulated_sval = (sval_total - sval_hat) / sval_total
                r = 0
                for ii in range(sval_ratio.shape[0]):
                    if accumulated_sval < threshold:
                        accumulated_sval += sval_ratio[ii]
                        r += 1
                    else:
                        break
                if r == 0:
                    feature = feature
                # update GPM
                U = np.hstack((feature, U[:, 0:r]))
                if U.shape[1] > U.shape[0]:
                    feature = U[:, 0:U.shape[0]]
                else:
                    feature = U
            print('-'*40)
            print('Gradient Constraints Summary', feature.shape)
            print('-'*40)
            features[layer][item] = feature

    return features

def row_cosine_similarity_sum(tensor1, tensor2):
    # 
    size1 = tensor1.shape[0]
    cosine_similarity = torch.nn.functional.cosine_similarity(tensor1, tensor2, dim=1)
    # 求和
    cosine_similarity_sum = torch.sum(cosine_similarity)
    return abs(cosine_similarity_sum.item() / size1)

def modify_available_list(task_id, available_mini_model_list, local_forward_type, init_with):
    print('----------------def modify_available_list----------------')
    print('>>> before modify:', available_mini_model_list)
    if task_id == 0:
        # task#1
        print('>>> : task0')
        available_mini_model_list[task_id] = task_id
    else:
        print('>>> : task{}'.format(task_id))
        if init_with is None and local_forward_type == 'add':
            available_mini_model_list[task_id] = task_id
        elif init_with is not None and local_forward_type == 'update':
            # print(1, available_mini_model_list)
            available_mini_model_list[task_id] = init_with
            # print(2, available_mini_model_list)

            new_dict = {}
            for key, value in available_mini_model_list.items():
                if value == init_with:

                    new_dict[key] = task_id
                else:
                    new_dict[key] = value
                    
            available_mini_model_list = new_dict

            
        else:
            pass
    print('>>> after modify:', available_mini_model_list)
    print('----------------def modify_available_list----------------')
    return available_mini_model_list

def get_dict_topk_max(input_dict, topk):

    dict_len = len(input_dict)
    if dict_len < topk:
        topk = dict_len

    value_to_key = {value: key for key, value in input_dict.items()}
    top_k_values = heapq.nsmallest(topk, value_to_key.keys(), value_to_key.get)

    top_k_keys = []
    for item in top_k_values:
        top_k_keys.append(value_to_key[item])

    return top_k_keys, top_k_values

def fill_with_enhance_id(idx, enhance_id):

    enhance_id_len = len(enhance_id)
    device = idx.device
    all_liked_idxs = []
    for item in range(enhance_id_len):
        liked_idx = torch.full_like(idx, fill_value=enhance_id[item]).to(device)
        all_liked_idxs.append(liked_idx)
    
    return all_liked_idxs

def file_exists(file_path):
    return os.path.exists(file_path)

def get_pretrained_feature_and_space(file_path, data_loader, device, original_model, class_num):


    upper_num = 10

    pre_feature_dict = {i: [] for i in range(class_num)}
    for pre_task_id in range(10):
        print('>>> processing on pre_task: {}'.format(pre_task_id))
        for img, target in data_loader[pre_task_id]['sample']:
            
            sub_target = int(target[0].cpu().numpy())
            if len(pre_feature_dict[sub_target]) == upper_num:
                pass
            else:
                input = img.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)

                with torch.no_grad():
                    if original_model is not None:
                        output = original_model(input)
                        # cls_features = output['pre_logits']
                        layer_feats = output['layer_feats']
                        layer_feats = torch.stack(layer_feats).squeeze() # [12, bs, seq, channel]
                        layer_feats = layer_feats.reshape(layer_feats.shape[0], -1)

                        pre_feature_dict[sub_target].append(layer_feats)

                        del img, target, output, layer_feats, sub_target

    pretrained_rep = pre_feature_dict

    new_pretrained_rep = {i:[] for i in range(100)}
    for key, value in pretrained_rep.items():
        pretrained_rep[key] = torch.stack(value)
    
    new_pretrained_rep_dict = {i:[] for i in range(12)}
    for key, value in pretrained_rep.items():
        for i in range(12):
            new_pretrained_rep_dict[i].append(value[:, i, :].detach().cpu()) # 11 ->  i

    for sub_key, sub_value in new_pretrained_rep_dict.items():
        print('>>> pca on sub_key', sub_key)
        value = torch.cat(sub_value)
        # rep = representation_e[layer][item]
        pca = PCA(n_components=768)
        pca = pca.fit(value)
        value = pca.transform(value)
        new_pretrained_rep_dict[sub_key] = {'key': value.T}
    
    with open(file_path, 'wb') as f:
        pickle.dump(new_pretrained_rep_dict, f)

    return new_pretrained_rep_dict
    

def pars_preparing(args, device, data_loader, original_model, class_num):
    file_path = './auto/pickle_data/{}/{}_pca.pickle'.format(args.original_model, args.dataset_name)
    if file_exists(file_path):
        print('>>> pretrained data exists')
        with open(file_path, 'rb') as f:
            new_pretrained_rep_dict = pickle.load(f)
    else:
        print('>>> build pretrained data')
        new_pretrained_rep_dict = get_pretrained_feature_and_space(file_path, data_loader, device, original_model, class_num)

    print(new_pretrained_rep_dict.keys())
    pretrained_subspace = update_memory_prefix(new_pretrained_rep_dict, threshold=args.threshold_pretrained, features={}) 
    print(pretrained_subspace.keys())
    pretrained_subspace_g = {i:{} for i in range(12)} 
    for layer in pretrained_subspace:
        for item in pretrained_subspace[layer]:
            temp_feature = pretrained_subspace[layer][item]
            Uf = torch.Tensor(np.dot(temp_feature, temp_feature.transpose())).to(device)
            print('e', layer, item, Uf.size())
            pretrained_subspace_g[layer][item] = Uf
            print("item", item)

    return pretrained_subspace, pretrained_subspace_g