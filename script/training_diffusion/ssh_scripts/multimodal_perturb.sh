MODEL_FLAGS="--cross_attention_resolutions 2,4,8  --cross_attention_windows 1,4,8 
--cross_attention_shift True  --use_cfgen True
--video_size 1,64 --audio_size 1,64 --learn_sigma False --num_channels 96
--num_head_channels -1 --num_res_blocks 1 --resblock_updown True --use_fp16 False
--use_scale_shift_norm True --class_cond True --condition cell_type"

# Modify --devices according your GPU number
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear
--all_save_num 2000 --save_type mp4  --devices 0, --classifier_scale 3.0
--batch_size 3000   --is_strict True --sample_fn ddpm"

# Modify the following paths to your own paths
MULTIMODAL_MODEL_PATH="/stor/lep/workspace/multi_diffusion/MM-Diffusion/outputs/checkpoints_cross/open_perturb_lr1e4_w384_scale124_80w_uncondi_wd1e4/model600000.pt"
AE_PATH="your/AE/path/checkpoints/last.ckpt"
OUT_DIR="/stor/lep/workspace/scDiffusion-X/script/training_diffusion/outputs/sample/my_perturb"  

# translation config
DATA_DIR="/stor/lep/diffusion/multiome/openproblem_filtered4perturb.h5mu"   # the translation source file
GEN_MODE="atac"     # the target modality
CONDITION="cell_type"  # the condition type to guide the generation
encoder_config="/stor/lep/workspace/scDiffusion-X/script/training_autoencoder/configs/encoder/encoder_multimodal_small.yaml"    # the autoencoder config
target_gene="CD3E,CD4"  # which genes to perturb. use comma to seperate.

NUM_GPUS=1
# mpiexec -n $NUM_GPUS 
CUDA_VISIBLE_DEVICES=0 python3 ../py_scripts/multimodal_perturb.py  \
$MODEL_FLAGS $SRMODEL_FLAGS $DIFFUSION_FLAGS $SR_DIFFUSION_FLAGS \
--output_dir ${OUT_DIR} --multimodal_model_path ${MULTIMODAL_MODEL_PATH} --data_dir ${DATA_DIR} --gen_mode ${GEN_MODE} \
--condition ${CONDITION} --encoder_config ${encoder_config} --ae_path ${AE_PATH} --target_gene ${target_gene}
