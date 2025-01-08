

MODEL_FLAGS="--cross_attention_resolutions 2,4,8  --cross_attention_windows 1,4,8 
--cross_attention_shift True  --video_attention_resolutions -1
--audio_attention_resolutions -1 --use_cfgen True
--video_size 1,64 --audio_size 1,64 --learn_sigma False --num_channels 96
--num_head_channels -1 --num_res_blocks 1 --resblock_updown True --use_fp16 False
--use_scale_shift_norm True --class_cond True --condition cell_type"
#   --num_class 22
SRMODEL_FLAGS="--sr_attention_resolutions 8,16,32  --large_size 256  
--small_size 64 --sr_learn_sigma True 
--sr_num_channels 192 --sr_num_heads 4 --sr_num_res_blocks 2 
--sr_resblock_updown True --use_fp16 False --sr_use_scale_shift_norm True"

# Modify --devices according your GPU number
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear
--all_save_num 2000 --save_type mp4  --devices 0, --classifier_scale 3.0
--batch_size 3000   --is_strict True --sample_fn ddpm"

SR_DIFFUSION_FLAGS="--sr_diffusion_steps 1000  --sr_sample_fn dpm_solver++ " # --sr_timestep_respacing ddim25

# Modify the following paths to your own paths
MULTIMODAL_MODEL_PATH="/stor/lep/workspace/multi_diffusion/MM-Diffusion/outputs/checkpoints_cross/open_perturb_lr1e4_w384_scale124_80w_uncondi_wd1e4/model600000.pt" #babel_lr1e4_w640_scale124_drop0_80w_rescale10_3crossatt64_layernorm
AE_PATH="/stor/lep/workspace/multi_diffusion/CFGen/project_folder/experiments/train_autoencoder_openproblem_multimodal_perturb/checkpoints/last.ckpt"
OUT_DIR="/stor/lep/workspace/multi_diffusion/MM-Diffusion/outputs/samples_trans/open_uncondi_layernorm_pert/80w_rna2atac_CD3ECD4_top5_x5"  # 80w_atac2rna_test_grad3   2 bs 2005, 3 bs 2000, 4 bs 3065, 5&ctrl2 fix ae input, ctrl3 bs 2000

# translation config
DATA_DIR="/stor/lep/diffusion/multiome/openproblem_filtered4perturb.h5mu"  #"/stor/lep/data/BABEL/train_celltype.h5mu"   # the translation source file
GEN_MODE="atac"     # the target modality
CONDITION="cell_type"  # the condition type to guide the generation
# NUM_CLASS="33"      # the number of classes in the dataset
encoder_config="encoder_multimodal_small"    # the cfgen autoencoder config

NUM_GPUS=1
# mpiexec -n $NUM_GPUS 
CUDA_VISIBLE_DEVICES=5 python3 py_scripts/multimodal_perturb.py  \
$MODEL_FLAGS $SRMODEL_FLAGS $DIFFUSION_FLAGS $SR_DIFFUSION_FLAGS \
--output_dir ${OUT_DIR} --multimodal_model_path ${MULTIMODAL_MODEL_PATH} --data_dir ${DATA_DIR} --gen_mode ${GEN_MODE} \
--condition ${CONDITION} --encoder_config ${encoder_config} --ae_path ${AE_PATH} #--num_class ${NUM_CLASS} 
