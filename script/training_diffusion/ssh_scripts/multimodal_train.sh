MODEL_FLAGS="--cross_attention_resolutions 2,4,8 --cross_attention_windows 1,4,8
--cross_attention_shift True --dropout 0.0 
--rna_dim 1,100 --atac_dim 1,100 --learn_sigma False --num_channels 128
--num_head_channels -1 --num_res_blocks 1 --resblock_updown True --use_fp16 False
--use_scale_shift_norm True --num_workers 4 --condition cell_type
--encoder_config /stor/lep/workspace/scDiffusion-X/script/training_autoencoder/configs/encoder/encoder_multimodal.yaml
--num_class 22 --weight_decay 0.0001
--ae_path /stor/lep/workspace/scDiffusion-X/script/training_autoencoder/outputs/checkpoints/my_autoencoder/checkpoints/last.ckpt
"

# Modify --devices to your own GPU ID
TRAIN_FLAGS="--lr 0.0001 --batch_size 128
--devices 0 --log_interval 100 --save_interval 200000 --use_db False --lr_anneal_steps=800000" 
DIFFUSION_FLAGS="--noise_schedule linear --diffusion_steps 1000 --sample_fn dpm_solver++" 

# Modify the following pathes to your own paths
DATA_DIR="/stor/lep/diffusion/multiome/openproblem_filtered.h5mu"
OUTPUT_DIR="/stor/lep/workspace/scDiffusion-X/script/training_diffusion/outputs/checkpoints/my_dfbackbone"
NUM_GPUS=1
WORLD_SIZE=1
NCCL_P2P_DISABLE=1

mpiexec -n $NUM_GPUS  python3 py_scripts/multimodal_train.py --data_dir ${DATA_DIR} --output_dir ${OUTPUT_DIR} $MODEL_FLAGS $TRAIN_FLAGS $VIDEO_FLAGS $DIFFUSION_FLAGS 
