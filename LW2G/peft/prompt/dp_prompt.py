import torch
import torch.nn as nn
from auto.common_utils import fill_with_enhance_id

class EPrompt(nn.Module):
    def __init__(self, length=5, embed_dim=768, embedding_key='mean', prompt_init='uniform', prompt_pool=False, 
                 prompt_key=False, pool_size=None, top_k=None, batchwise_prompt=False, prompt_key_init='uniform',
                 num_layers=1, use_prefix_tune_for_e_prompt=False, num_heads=-1, same_key_value=False,):
        super().__init__()

        self.length = length
        self.prompt_pool = prompt_pool
        self.embedding_key = embedding_key
        self.prompt_init = prompt_init
        self.prompt_key = prompt_key
        self.pool_size = pool_size
        self.top_k = top_k
        self.batchwise_prompt = batchwise_prompt
        self.num_layers = num_layers
        self.use_prefix_tune_for_e_prompt = use_prefix_tune_for_e_prompt
        self.num_heads = num_heads
        self.same_key_value = same_key_value

        if self.prompt_pool:
            # user prefix style
            if self.use_prefix_tune_for_e_prompt:
                assert embed_dim % self.num_heads == 0
                if self.same_key_value:
                    prompt_pool_shape = (self.num_layers, 1, self.pool_size, self.length, 
                                        self.num_heads, embed_dim // self.num_heads)

                    if prompt_init == 'zero':
                        self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
                    elif prompt_init == 'uniform':
                        self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
                        nn.init.uniform_(self.prompt, -1, 1)
                    self.prompt = self.prompt.repeat(1, 2, 1, 1, 1, 1)
                else:
                    prompt_pool_shape = (self.num_layers, 2, self.pool_size, self.length, 
                                        self.num_heads, embed_dim // self.num_heads)
                    if prompt_init == 'zero':
                        self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
                    elif prompt_init == 'uniform':
                        self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
                        nn.init.uniform_(self.prompt, -1, 1)
            else:
                prompt_pool_shape = (self.num_layers, self.pool_size, self.length, embed_dim)  # TODO fix self.num_layers = 1
                if prompt_init == 'zero':
                    self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
                elif prompt_init == 'uniform':
                    self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
                    nn.init.uniform_(self.prompt, -1, 1)
                    
        # if using learnable prompt keys
        if prompt_key:
            key_shape = (pool_size, embed_dim)
            if prompt_key_init == 'zero':
                self.prompt_key = nn.Parameter(torch.zeros(key_shape))
            elif prompt_key_init == 'uniform':
                self.prompt_key = nn.Parameter(torch.randn(key_shape))
                nn.init.uniform_(self.prompt_key, -1, 1)
        else:
            # else use mean of prompt as key
            # only compatible with prompt, not prefix
            prompt_mean = torch.mean(self.prompt, dim=[0, 2])
            self.prompt_key = prompt_mean 
            
    def l2_normalize(self, x, dim=None, epsilon=1e-12):
        """Normalizes a given vector or matrix."""
        square_sum = torch.sum(x ** 2, dim=dim, keepdim=True)
        x_inv_norm = torch.rsqrt(torch.maximum(square_sum, torch.tensor(epsilon, device=x.device)))
        return x * x_inv_norm
    
    def forward(self, x_embed, prompt_mask=None, cls_features=None, 
                prompt_momentum=0, available_mini_model_list=None, training=False,
                enhance_id=None):
        
        out = dict()
        if cls_features == 'only_sampling':
            idx = prompt_mask          
            out['prompt_idx'] = idx

            batched_prompt_raw = self.prompt[:, :, idx]
            
            batched_prompt_raw = batched_prompt_raw.permute(0, 2, 1, 3, 4, 5, 6)

            num_layers, batch_size, dual, top_k, length, num_heads, heads_embed_dim = batched_prompt_raw.shape
            batched_prompt = batched_prompt_raw.reshape(
                num_layers, batch_size, dual, top_k * length, num_heads, heads_embed_dim
            )
   
            out['batched_prompt'] = batched_prompt
            
        else:
            if self.prompt_pool:
                if self.embedding_key == 'mean':
                    x_embed_mean = torch.mean(x_embed, dim=1)
                elif self.embedding_key == 'max':
                    x_embed_mean = torch.max(x_embed, dim=1)[0]
                elif self.embedding_key == 'mean_max':
                    x_embed_mean = torch.max(x_embed, dim=1)[0] + 2 * torch.mean(x_embed, dim=1)
                elif self.embedding_key == 'cls':
                    if cls_features is None:
                        x_embed_mean = torch.max(x_embed, dim=1)[0] # B, C
                    else:
                        x_embed_mean = cls_features
                else:
                    raise NotImplementedError("Not supported way of calculating embedding keys!")

                prompt_key_norm = self.l2_normalize(self.prompt_key, dim=-1) # Pool_size, C
                x_embed_norm = self.l2_normalize(x_embed_mean, dim=-1) # B, C

                similarity = torch.matmul(prompt_key_norm, x_embed_norm.t()) # pool_size, B or Pool_size, #class, B
                similarity = similarity.t() # B, pool_size
                (similarity_top_k, idx) = torch.topk(similarity, k=self.top_k, dim=1) # B, top_k
                
                if training:
                    pass
                else:
                    idx = idx.cpu()
                    idx = idx.apply_(lambda x: available_mini_model_list[x] if isinstance(x, int) and x in available_mini_model_list else available_mini_model_list[x.item()] if isinstance(x, torch.Tensor) and x.item() in available_mini_model_list else x)

                    idx = idx.to(similarity_top_k.device) 
                                
                out['similarity'] = similarity

                if self.batchwise_prompt:
                    prompt_id, id_counts = torch.unique(idx, return_counts=True, sorted=True)
                    # In jnp.unique, when the 'size' is specified and there are fewer than the indicated number of elements,
                    # the remaining elements will be filled with 'fill_value', the default is the minimum value along the specified dimension.
                    # Unless dimension is specified, this will be flattend if it is not already 1D.
                    if prompt_id.shape[0] < self.pool_size:
                        prompt_id = torch.cat([prompt_id, torch.full((self.pool_size - prompt_id.shape[0],), torch.min(idx.flatten()), device=prompt_id.device)])
                        id_counts = torch.cat([id_counts, torch.full((self.pool_size - id_counts.shape[0],), 0, device=id_counts.device)])
                    _, major_idx = torch.topk(id_counts, k=self.top_k) # top_k
                    major_prompt_id = prompt_id[major_idx] # top_k
                    # expand to batch
                    idx = major_prompt_id.expand(x_embed.shape[0], -1).contiguous() # B, top_k
                
                #10. enhance id
                if prompt_mask is not None:
                    idx = prompt_mask # B, top_k
                    if enhance_id is not None:
                        enhance_idxs = fill_with_enhance_id(idx=idx, enhance_id=enhance_id)
                    else:
                        pass

                out['prompt_idx'] = idx
                if self.use_prefix_tune_for_e_prompt:
                    if prompt_momentum > 0 and prompt_mask is not None:
                        with torch.no_grad():
                            batched_prompt_momentum = self.prompt[:, :, idx-1].detach().clone()
                            batched_prompt_raw = (1-prompt_momentum) * self.prompt[:, :, idx] + prompt_momentum * batched_prompt_momentum
                        self.prompt[:, :, idx].copy_(batched_prompt_raw)
                        # batched_prompt_raw = self.prompt[:, :, idx]
                        num_layers, dual, batch_size, top_k, length, num_heads, heads_embed_dim = batched_prompt_raw.shape
                        batched_prompt = batched_prompt_raw.reshape(
                            num_layers, batch_size, dual, top_k * length, num_heads, heads_embed_dim
                        )
                    else:
                        batched_prompt_raw = self.prompt[:, :, idx]  # num_layers, B, top_k, length, C
                        
                        batched_prompt_raw = batched_prompt_raw.permute(0, 2, 1, 3, 4, 5, 6)

                        num_layers, batch_size, dual, top_k, length, num_heads, heads_embed_dim = batched_prompt_raw.shape
                        batched_prompt = batched_prompt_raw.reshape(
                            num_layers, batch_size, dual, top_k * length, num_heads, heads_embed_dim
                        )
                    
                    if enhance_id is not None:
                        for sub_idx in enhance_idxs:
                            # no grad but only forward
                            p_raw = self.prompt[:, :, sub_idx].detach().clone()  #

                            p_raw = p_raw.permute(0, 2, 1, 3, 4, 5, 6)

                            num_layers, batch_size, dual, top_k, length, num_heads, heads_embed_dim = p_raw.shape
                            batched_p = p_raw.reshape(
                                num_layers, batch_size, dual, top_k * length, num_heads, heads_embed_dim
                            )
                            # concat on batched_prompt(dim=0)
                            batched_prompt = torch.cat((batched_prompt, batched_p), dim=0)
                    else:
                        pass
                else:
                    batched_prompt_raw = self.prompt[:, idx]
                    num_layers, batch_size, top_k, length, embed_dim = batched_prompt_raw.shape
                    batched_prompt = batched_prompt_raw.reshape(
                        num_layers, batch_size, top_k * length, embed_dim
                    )

                batched_key_norm = prompt_key_norm[idx] # B, top_k, C

                out['selected_key'] = batched_key_norm
                out['prompt_key_norm'] = prompt_key_norm
                out['x_embed_norm'] = x_embed_norm

                # Put pull_constraint loss calculation inside
                x_embed_norm = x_embed_norm.unsqueeze(1) # B, 1, C
                sim = batched_key_norm * x_embed_norm # B, top_k, C
                reduce_sim = torch.sum(sim) / x_embed.shape[0] # Scalar
                
                out['reduce_sim'] = reduce_sim
            else:
                # user prefix style
                if self.use_prefix_tune_for_e_prompt:
                    assert embed_dim % self.num_heads == 0
                    if self.same_key_value:
                        prompt_pool_shape = (self.num_layers, 1, self.length, 
                                            self.num_heads, embed_dim // self.num_heads)
                        if self.prompt_init == 'zero':
                            self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
                        elif self.prompt_init == 'uniform':
                            self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
                            nn.init.uniform_(self.prompt, -1, 1)
                        self.prompt = self.prompt.repeat(1, 2, 1, 1, 1)
                    else:
                        prompt_pool_shape = (self.num_layers, 2, self.length, 
                                            self.num_heads, embed_dim // self.num_heads)
                        if self.prompt_init == 'zero':
                            self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
                        elif self.prompt_init == 'uniform':
                            self.prompt = nn.Parameter(torch.randn(prompt_pool_shape)) # num_layers, 2, length, num_heads, embed_dim // num_heads
                            nn.init.uniform_(self.prompt, -1, 1)
                    batched_prompt = self.prompt.unsqueeze(0).expand(-1, x_embed.shape[0], -1, -1, -1)  # TODO
                else:
                    prompt_pool_shape = (self.num_layers, self.length, embed_dim)
                    if self.prompt_init == 'zero':
                        self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
                    elif self.prompt_init == 'uniform':
                        self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
                        nn.init.uniform_(self.prompt, -1, 1)
                    batched_prompt = self.prompt.unsqueeze(0).expand(-1, x_embed.shape[0], -1, -1)  # TODO
            
            out['batched_prompt'] = batched_prompt

        return out
