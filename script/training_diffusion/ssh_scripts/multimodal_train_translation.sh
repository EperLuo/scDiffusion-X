
#################64 x 64 uncondition###########################################################
MODEL_FLAGS="--cross_attention_resolutions 2,4,8 --cross_attention_windows 1,4,8
--cross_attention_shift True --dropout 0.0 
--video_size 1,150 --audio_size 1,200 --learn_sigma False --num_channels 160
--num_head_channels -1 --num_res_blocks 1 --resblock_updown True --use_fp16 False
--use_scale_shift_norm True --num_workers 4 --condition leiden
--encoder_config /stor/lep/workspace/scMulDiffusion/script/training_autoencoder/configs/encoder/encoder_multimodal_large.yaml
--weight_decay 0.0001
--ae_path your/AE/path/checkpoints/last.ckpt
"

# Modify --devices to your own GPU ID
TRAIN_FLAGS="--lr 0.0001 --batch_size 64
--devices 0,1 --log_interval 100 --save_interval 200000 --use_db False --lr_anneal_steps=800000"
DIFFUSION_FLAGS="--noise_schedule linear --diffusion_steps 1000 --sample_fn dpm_solver++" 

# Modify the following pathes to your own paths
DATA_DIR="/stor/lep/data/BABEL/train_celltype.h5mu" 
OUTPUT_DIR="/stor/lep/workspace/scMulDiffusion/script/training_diffusion/outputs/checkpoints/my_dfbackbone2"
NUM_GPUS=2
WORLD_SIZE=1
NCCL_P2P_DISABLE=1

mpiexec -n $NUM_GPUS  python3 ../py_scripts/multimodal_train.py --data_dir ${DATA_DIR} --output_dir ${OUTPUT_DIR} $MODEL_FLAGS $TRAIN_FLAGS $VIDEO_FLAGS $DIFFUSION_FLAGS 
