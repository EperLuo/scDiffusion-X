MODEL_FLAGS="--cross_attention_resolutions 2,4,8  --cross_attention_windows 1,4,8 
--cross_attention_shift True
--video_size 1,150 --audio_size 1,200 --learn_sigma False --num_channels 160
--num_head_channels -1 --num_res_blocks 1 --resblock_updown True --use_fp16 False
--use_scale_shift_norm True --class_cond True"

# Modify --devices according your GPU number
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear
--devices 0, --classifier_scale 3.0
--batch_size 2000   --is_strict True --sample_fn ddpm"

# Modify the following paths to your own paths
MULTIMODAL_MODEL_PATH="/stor/lep/workspace/multi_diffusion/MM-Diffusion/outputs/checkpoints_ablation/babel_lr1e4_w640_scale124_drop0_80w_rescale10_3crossatt64_layernorm2/model800000.pt"
AE_PATH="your/AE/path/checkpoints/last.ckpt"
OUT_DIR="/stor/lep/workspace/scDiffusion-X/script/training_diffusion/outputs/sample/my_translation" 

# translation config
DATA_DIR="/stor/lep/data/BABEL/test.h5mu"  # the source data
GEN_MODE="rna"     # the target modality
CONDITION="leiden"  # the condition type to guide the generation
encoder_config="/stor/lep/workspace/scDiffusion-X/script/training_autoencoder/configs/encoder/encoder_multimodal_large.yaml"    # the cfgen autoencoder config
gen_times="5"    # how many time you want to translate the whole data. usually translate more than once to remove noise.

NUM_GPUS=1
# mpiexec -n $NUM_GPUS 
CUDA_VISIBLE_DEVICES=1 python3 ../py_scripts/multimodal_translation.py  \
$MODEL_FLAGS $SRMODEL_FLAGS $DIFFUSION_FLAGS $SR_DIFFUSION_FLAGS \
--output_dir ${OUT_DIR} --multimodal_model_path ${MULTIMODAL_MODEL_PATH} --data_dir ${DATA_DIR} --gen_mode ${GEN_MODE} \
--condition ${CONDITION} --encoder_config ${encoder_config} --ae_path ${AE_PATH} --gen_times ${gen_times} 
