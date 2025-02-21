## scDiffusion-X: Diffusion Model for Single-Cell Multiome Data Generation and Analysis

Welcome! This is the official implement of scDiffusion-X.

TODO: introduction to scDiffusion-X
<!-- ![image](FIG1.png) -->
<div align="center">  
    <img src="FIG1.png" width="650">  
</div>  

# Installation
<!-- Use conda create:
```
conda create --name scmuldiff --file requirements.txt python=3.8
```
Use setup.py:

First clone this repository into your local path. Then run:
```
cd scDiffusion-X
pip install -e .
```
TODO: Pipy package construction -->
```
conda create --name scmuldiff python=3.8
pip install -r requirements.txt
pip install scdiffusionX
conda install mpi4py
```


# User guidance

**Step1: Train the Autoencoder**
```
cd script/training_autoencoder
bash train_autoencoder_multimodal.sbatch
```
Adjust the data path to your local path. The dataset config file is in script/training_autoencoder/configs/dataset, see the comments in openproblem.yaml for details. The checkpoint will be saved in script/training_autoencoder/outputs/checkpoints and the log file will be saved in script/training_autoencoder/outputs/logs. The autoencoder config file is in script/training_autoencoder/configs/encoder, see the comments in encoder_multimodal.yaml for details. 

We recommand to use encoder_multimodal for most of dataset. If the genes and peaks are more than 50,000 and 200,000, we recommand a larger autoencoder in encoder_multimodal_large. If the genes and peaks are less than 5,000 and 15,000, we recommand a smaller autoencoder in encoder_multimodal_small. The `norm_type` in the encoder config yaml control the normalization type. For data generation task, we recommend batch_norm, and for translation task, we recommend layer_norm since it has better generalization for OOD data.

**Step2: Train the Diffusion Backbone**

```
cd script/training_diffusion
sh ssh_scripts/multimodal_train.sh
```
Again, adjust the data path and output path to your own, and also change the ae_path&encoder_config to the autoencoder you tarined in step 1. When training with condition (like the cell type condition), set the `num_class` to the number of unique labels. The training is unconditional when the `num_class` is not set.

TODO: Explain more about each attribution

**Step3: Generate new data**

```
cd script/training_diffusion
sh ssh_scripts/multimodal_sample.sh
```
Change the MULTIMODAL_MODEL_PATH to the checkpoint path in step 2, and the DATA_DIR to your local data path.

The experiments results in the paper can be reproduce through `evaluate_script/inference_multi_diff.ipynb`

TODO: More details about the hyperpara, conditional and unconditional

**Founction: Modality translation**

For this task, we recommend you use `layer_norm` instead of `batch_norm` since it fit more for the OOD data. And if your source modality doesn't have a condition label overlap with the training data (like a external dataset), you can use unconditional training to train the model. If so, use a clustering method like leiden to get the cluster label as the covariate_keys for encoder (to get the size factor).
```
cd script/training_diffusion
sh ssh_scripts/multimodal_train_translation.sh
sh ssh_scripts/multimodal_translation.sh
```
You need to change the file path in both bash file to your local path. The `GEN_MODE` is the target modality (either "rna" or "atac" for current model). The training logic is the same for the multimodal_train_translation.sh and multimodal_train.sh except the dataset and other hyperparameters.

The experiments results in the paper can be reproduce through `evaluate_script/translation_multi_diff.ipynb`

TODO: change the format of input data file. More explaination about the hyperparameters and setting.

**Founction: Gene-Peak regulatory analysis**

You need to first complete the step1 and step2. The detail implement can be found in ``evaluate_script/regulatory_multi_diff.ipynb``

<!-- Acknowledge: the code of this project is based on CFGen:https://github.com/theislab/CFGen and MM-diffusion: https://github.com/researchmm/MM-Diffusion. -->