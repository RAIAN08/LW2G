<div align="center">
  
  <div>
  <h1>LW2G: Learning Whether to Grow for Prompt-based Continual Learning</h1>
  </div>

  <div>
  </div>
  <br/>

</div>



![illustration_figure](main_pic.png)

Continual Learning (CL) aims to learn in non-stationary scenarios, progressively acquiring and maintaining knowledge from sequential tasks. The recent Prompt-based Continual Learning (PCL) approaches have achieved remarkable performance by leveraging the Pre-Trained Models (PTMs). These approaches grow the pool by adding a new set of prompts when learning each new task (prompt learning) and adopt a matching mechanism to select the correct set for each testing sample (prompt retrieval). Previous studies make efforts on improving the matching mechanism to enhance Prompt Retrieval Accuracy (PRA). Inspired by humans' integration of information, we propose to dynamically learn whether to grow a new set of prompts during prompt learning based on the disparities between tasks, rather than crudely growing for each task. Specifically, when several tasks share certain commonalities, they can utilize a shared set of prompts. Conversely, when tasks exhibit significant differences, a new set should be added. In this paper, we propose a plug-in module with PCL to Learn Whether to Grow (LW2G). In LW2G, we surgically modify the gradients of the new task with Gradient Projection Continual Learning (GPCL), thereby encoding the knowledge from multiple tasks into a single set of prompts (opting not to grow). Furthermore, we introduce a metric called Hinder Forward Capability (HFC) to measure the hindrance imposed on learning new tasks under the strict orthogonal condition in GPCL, while also quantifying the disparities between new and old tasks. Hence, when learning new tasks is severely impeded, it is warranted to dynamically opt to grow a new set. Extensive experiments show the effectiveness of our method. 


# Note
We first need to explain that HidePrompt is a two-stage method. In the first stage, it trains an adapter for retrieval. In the second stage, it gradually increases the set of prompts for each task. Since LW2G does not make any changes to the first stage but acts as a plug-in module in the second stage, we choose to reuse HidePrompt's official code to generate the checkpoint for the first stage. The generation process strictly follows HidePrompt's official code.

Therefore, to reproduce HidePrompt [+ LW2G] and its baseline on each benchmarks, we must first execute the following three steps to fetch the checkpoint from the first stage, and then proceed with the LW2G training.

For DualPrompt and S-Prompt++, since they are end-to-end training methods, we can directly train LW2G.

# CIFAR

### 1 To reproduced the results of HidePrompt [ + LW2G] and its coresponing baseline HidePrompt on CIFAR.

1. `cd HiDe-Prompt-main_after_modified/`
2. `bash new_bash/hideprompt_cifar_prepare.bash`
3. `mv ckpt_for_hidep/ ../LW2G`
4. `cd ../LW2G`
5. `bash new_bash/cifar/hideprompt_lw2g.bash` and `bash new_bash/cifar/hideprompt.bash`



### 2 To reproduced the results of DualPrompt [ + LW2G] and its coresponing baseline DualPrompt on CIFAR.

1. `cd LW2G/`
2. `bash new_bash/cifar/dualprompt_lw2g.bash` and `bash new_bash/cifar/dualprompt.bash`



# IMR

### 1 To reproduced the results of HidePrompt [ + LW2G] and its coresponing baseline HidePrompt on IMR.

1. `cd HiDe-Prompt-main_after_modified/`
2. `bash new_bash/hideprompt_imr_prepare.bash`
3. `mv ckpt_for_hidep/ ../LW2G`
4. `cd ../LW2G`
5. `bash new_bash/imr/hideprompt_lw2g.bash` and `bash new_bash/imr/hideprompt.bash`

### 2 To reproduced the results of DualPrompt [ + LW2G] and its coresponing baseline DualPrompt on IMR.

1. `cd LW2G/`
2. `bash new_bash/imr/dualprompt_lw2g.bash` and `bash new_bash/imr/dualprompt.bash`

# CUB

### 1 To reproduced the results of HidePrompt [ + LW2G] and its coresponing baseline HidePrompt on CUB.

1. `cd HiDe-Prompt-main_after_modified/`
2. `bash new_bash/hideprompt_cub_prepare.bash`
3. `mv ckpt_for_hidep/ ../LW2G`
4. `cd ../LW2G`
5. `bash new_bash/cub/hideprompt_lw2g.bash` and `bash new_bash/cub/hideprompt.bash`

### 2 To reproduced the results of DualPrompt [ + LW2G] and its coresponing baseline DualPrompt on CUB.

1. `cd LW2G/`
2. `bash new_bash/cub/dualprompt_lw2g.bash` and `bash new_bash/cub/dualprompt.bash`

# Logs

Due to potential factors such as the experimental environment and CUDA version, as well as variations in configurations when using DDP for multi-node, multi-GPU training in the first stage of HidePrompt, the resulting checkpoints may differ. Therefore, we have provided all training logs at:

`
./LW2G/all_logs/
`

# Envs

Please refer to `./issue2/LW2G/requirements.txt`.
