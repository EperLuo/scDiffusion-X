MODEL_FLAGS="--cross_attention_resolutions 2,4,8  --cross_attention_windows 1,4,8 
--cross_attention_shift True
--rna_dim 1,100 --atac_dim 1,100 --learn_sigma False --num_channels 128 
--num_head_channels -1 --num_res_blocks 1 --resblock_updown True --use_fp16 False
--use_scale_shift_norm True  --class_cond True --condition cell_type --num_class 22"
# --specific_type 1

# Modify --devices according your GPU number
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear
--all_save_num 2000  --devices 6
--batch_size 2000  --is_strict True --sample_fn ddim"

# Modify the following paths to your own paths
MULTIMODAL_MODEL_PATH="/stor/lep/workspace/scDiffusion-X/script/training_diffusion/outputs/checkpoints/my_backbone/model800000.pt"
OUT_DIR="/stor/lep/workspace/scDiffusion-X/script/training_diffusion/outputs/sample/my_sample"
DATA_DIR="/stor/lep/diffusion/multiome/openproblem_filtered.h5mu"

NUM_GPUS=1
mpiexec -n $NUM_GPUS python3 py_scripts/multimodal_sample.py  \
$MODEL_FLAGS $SRMODEL_FLAGS $DIFFUSION_FLAGS $SR_DIFFUSION_FLAGS \
--output_dir ${OUT_DIR} --multimodal_model_path ${MULTIMODAL_MODEL_PATH} --data_dir ${DATA_DIR}
