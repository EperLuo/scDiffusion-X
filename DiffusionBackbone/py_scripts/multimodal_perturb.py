"""
Generate a large batch of video-audio pairs
"""
import sys,os
sys.path.append(os.path.dirname (os.path.dirname (os.path.abspath (__file__))))
import argparse
import os
import numpy as np
import torch as th
import torch.distributed as dist
from einops import rearrange, repeat
import muon as mu
import yaml

from mm_diffusion import dist_util, logger
from mm_diffusion.multimodal_script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict
)
from mm_diffusion.script_util import (
    image_sr_model_and_diffusion_defaults,
    image_sr_create_model_and_diffusion
)
from mm_diffusion.common import set_seed_logger_random, save_audio, save_img, save_multimodal, delete_pkl
from mm_diffusion.multimodal_dpm_solver_plus import DPM_Solver as multimodal_DPM_Solver
from mm_diffusion.dpm_solver_plus import DPM_Solver as singlemodal_DPM_Solver
# from mm_diffusion.evaluator import eval_multimodal

sys.path.append('../Autoencoder')
from autoencoder.models.base.encoder_model import EncoderModel

def main():
    args = create_argparser().parse_args()
    args.video_size = [int(i) for i in args.video_size.split(',')]
    args.audio_size = [int(i) for i in args.audio_size.split(',')]
    
    
    dist_util.setup_dist(args.devices)
    logger.configure(args.output_dir)
    args = set_seed_logger_random(args)


    logger.log("creating model and diffusion...")
    multimodal_model, multimodal_diffusion = create_model_and_diffusion(
         **args_to_dict(args, [key for key in model_and_diffusion_defaults().keys()])
    )
    sr_model, sr_diffusion = image_sr_create_model_and_diffusion(
        **args_to_dict(args, [key for key in image_sr_model_and_diffusion_defaults().keys()])
    )

    if os.path.isdir(args.multimodal_model_path):
        multimodal_name_list = [model_name for model_name in os.listdir(args.multimodal_model_path) \
            if (model_name.startswith('model') and model_name.endswith('.pt') and int(model_name.split('.')[0][5:])>= args.skip_steps)]
        multimodal_name_list.sort()
        multimodal_name_list = [os.path.join(args.model_path, model_name) for model_name in multimodal_name_list[::1]]
    else:
        multimodal_name_list = [model_path for model_path in args.multimodal_model_path.split(',')]
        
    logger.log(f"models waiting to be evaluated:{multimodal_name_list}")

    if os.path.exists(args.sr_model_path):
        sr_model.load_state_dict_(
            dist_util.load_state_dict(args.sr_model_path, map_location="cpu"), is_strict=args.is_strict
        )
        sr_model.to(dist_util.dev())
        if args.use_fp16:
            sr_model.convert_to_fp16()
    else:
        sr_model = None

    sr_noise=None
    if os.path.exists(args.load_noise):
        sr_noise = np.load(args.load_noise)
        sr_noise = th.tensor(sr_noise).to(dist_util.dev()).unsqueeze(0)
        sr_noise = repeat(sr_noise, 'b c h w -> (b repeat) c h w', repeat=args.batch_size * args.video_size[0])
        if dist.get_rank()==0:
            logger.log(f"load noise form {args.load_noise}...")

    for model_path in multimodal_name_list:
        multimodal_model.load_state_dict_(
            dist_util.load_state_dict(model_path, map_location="cpu"), is_strict=args.is_strict
        )
        
        multimodal_model.to(dist_util.dev())
        if args.use_fp16:
            multimodal_model.convert_to_fp16()
        multimodal_model.eval()

        logger.log(f"sampling samples for {model_path}")
        model_name = model_path.split('/')[-1]

        groups= 0
        multimodal_save_path = os.path.join(args.output_dir, model_name, 'original')
        sr_save_path = os.path.join(args.output_dir, model_name, 'sr_mp4')
        audio_save_path = os.path.join(args.output_dir)
        img_save_path = os.path.join(args.output_dir)
        if dist.get_rank() == 0:
            os.makedirs(multimodal_save_path, exist_ok=True)
            os.makedirs(sr_save_path, exist_ok=True)
            os.makedirs(audio_save_path, exist_ok=True)
            os.makedirs(img_save_path, exist_ok=True)

        mdata = mu.read_h5mu(args.data_dir)
        if args.class_cond:
            from sklearn.preprocessing import LabelEncoder
            labels = mdata['rna'].obs[args.condition].values
            label_encoder = LabelEncoder()
            label_encoder.fit(labels)
            classes_all = label_encoder.transform(labels)
            
        mdata = mdata#[::5]
        classes_all = classes_all#[::5]
        
        with open(f'{os.path.dirname(os.path.realpath(__file__))}/../../CFGen/configs/configs_encoder/encoder/{args.encoder_config}.yaml', 'r') as file:
            yaml_content = file.read()
        autoencoder_args = yaml.safe_load(yaml_content)

        # autoencoder_args = {'encoder_kwargs': {'rna': {'dims': [768, 450, 150], 'batch_norm': True, \
        #             'dropout': False, 'dropout_p': 0.0}, 'atac': {'dims': [1536, 768, 150], \
        #             'batch_norm': True, 'dropout': False, 'dropout_p': 0.0}}, 'learning_rate': 0.001, \
        #             'weight_decay': 1e-05, 'covariate_specific_theta': False, 'multimodal': True, \
        #             'is_binarized': True, 'encoder_multimodal_joint_layers': None}

        # Initialize encoder                'atac': 143810, 'rna': 19448
        autoencoder_args['encoder_kwargs']['rna']['norm_type']='layernorm'
        autoencoder_args['encoder_kwargs']['atac']['norm_type']='layernorm'
        encoder_model = EncoderModel(in_dim={'atac': mdata['atac'].shape[1], 'rna': mdata['rna'].shape[1]},
                                            n_cat=label_encoder.classes_.shape[0],
                                            conditioning_covariate=args.condition, 
                                            encoder_type='learnt_autoencoder',
                                            **autoencoder_args)

        # Load weights 
        encoder_model.load_state_dict(th.load(args.ae_path)["state_dict"])
        encoder_model.to(dist_util.dev())
        # initialize the source modality
        batch = {}

        target_gene=['CD3E','CD4']    #'CD3E', 'CD3E,CD4', 'CD4', 'CTRL', 'NFKB2', 'ZAP70'
        target_index = np.where(np.in1d(mdata['rna'].var_names.values,target_gene)==1)[0]
        selected_cell_types = ['CD4+ T naive','CD4+ T activated', 'CD8+ T', 'CD8+ T naive'] #['naive CD4 T cells', 'memory CD4 T cells']
        # mdata['rna'][:, target_gene] = 0
        # mdata['rna'][mdata['rna'].obs['cell_type'].isin(selected_cell_types),target_gene] = 0

        gt_rna = th.tensor(mdata['rna'][mdata['rna'].obs['cell_type'].isin(selected_cell_types),:].X.toarray(),device=dist_util.dev())
        gt_atac = th.tensor(mdata['atac'][mdata['rna'].obs['cell_type'].isin(selected_cell_types),:].X.toarray(),device=dist_util.dev())
        
        select_count = int(gt_rna.shape[0] * 0.05)

        index = np.where(((np.array(gt_rna[:,target_index[0]].detach().cpu()>0).astype(np.int32) + np.array(gt_rna[:,target_index[1]].detach().cpu()>0)>1).astype(np.int32))==1)[0]
        gt_rna = gt_rna[index]
        gt_atac = gt_atac[index]        

        sorted_indices = th.argsort(gt_rna[:,target_index[0]], descending=True)
        top_indices = sorted_indices[:select_count].to(dist_util.dev())  
        gt_rna = gt_rna[top_indices]
        # gt_rna = (gt_rna>0)*5.0
        gt_atac = gt_atac[top_indices]
        print(gt_rna[:, target_index].shape)
        print(gt_rna[:, target_index])

        # if perturb
        gt_rna[:, target_index] = 0 

        batch["X_norm"] = {'rna':gt_rna,'atac':gt_atac}
        z = encoder_model.encode(batch)
        noise_init = z[next(s for s in z.keys() if s != args.gen_mode)]

        # noise_init = noise_init[mdata['rna'].obs['cell_type'].isin(selected_cell_types)]
        classes_all = classes_all[mdata['rna'].obs['cell_type'].isin(selected_cell_types)]

        npzfile = np.load('/'.join(args.ae_path.split('/')[:-2])+'/norm_factor.npz')
        std = npzfile['rna_std'] if args.gen_mode == 'atac' else npzfile['atac_std']
        noise_init = noise_init/th.tensor(std,device=noise_init.device)

        videos = []
        audios = []
        all_labels = []

        # while groups * args.batch_size *  dist.get_world_size()< args.all_save_num: 
        sample_num = noise_init.shape[0]
        num_iteration = int(sample_num/args.batch_size)+1
        for i in list(range(num_iteration))*5:
       
            model_kwargs = {}

            x_T_init = noise_init[i*args.batch_size:(i+1)*args.batch_size]
            # random_indices = th.randint(0, noise_init.size(0), (args.batch_size*5,))
            # selected_tensor = noise_init[random_indices]
            # x_T_init = selected_tensor.view(-1, 5, noise_init.shape[-1]).mean(dim=1)
            if args.gen_mode == 'rna':
                model_kwargs["audio"] = x_T_init.unsqueeze(1).to(dist_util.dev())
            else:
                model_kwargs["video"] = x_T_init.unsqueeze(1).to(dist_util.dev())
            if args.class_cond:
                classes = classes_all[i*args.batch_size:(i+1)*args.batch_size]  # generated random cell type
                # classes = th.ones(args.batch_size)*3   # generated certain cell type
                # classes = th.randint(
                #    low=0, high=args.num_class, size=(args.batch_size,), device=dist_util.dev()
                # )
                classes = th.tensor(classes, device=dist_util.dev(), dtype=th.int)
                model_kwargs["label"] = classes

            shape = {"video":(args.batch_size if i!=num_iteration-1 else x_T_init.shape[0], *args.video_size), \
                    "audio":(args.batch_size if i!=num_iteration-1 else x_T_init.shape[0], *args.audio_size)
                }
            if args.sample_fn == 'dpm_solver':
                # sample_fn = multimodal_dpm_solver
                # sample = sample_fn(shape = shape, \
                #     model_fn = multimodal_model, steps=args.timestep_respacing)

                dpm_solver = multimodal_DPM_Solver(model=multimodal_model, \
                    alphas_cumprod=th.tensor(multimodal_diffusion.alphas_cumprod, dtype=th.float32))
                x_T = {"video":th.randn(shape["video"]).to(dist_util.dev()), \
                        "audio":th.randn(shape["audio"]).to(dist_util.dev())}
                sample = dpm_solver.sample(
                    x_T,
                    steps=20,
                    order=3,
                    skip_type="logSNR",
                    method="singlestep",
                )

            elif args.sample_fn == 'dpm_solver++':
                dpm_solver = multimodal_DPM_Solver(model=multimodal_model, \
                    alphas_cumprod=th.tensor(multimodal_diffusion.alphas_cumprod, dtype=th.float32), \
                        predict_x0=True, thresholding=True)
                
                x_T = {"video":th.randn(shape["video"]).to(dist_util.dev()), \
                        "audio":th.randn(shape["audio"]).to(dist_util.dev())}
                sample = dpm_solver.sample(
                    x_T,
                    steps=20,
                    order=2,
                    skip_type="logSNR",
                    method="adaptive",
                )
            else:
                sample_fn = (
                    multimodal_diffusion.conditional_p_sample_loop if  args.sample_fn=="ddpm" else multimodal_diffusion.ddim_sample_loop
                )

                sample = sample_fn(
                    multimodal_model,
                    shape = shape,
                    clip_denoised=args.clip_denoised,
                    model_kwargs=model_kwargs,
                    # noise=x_T_init,
                    # gen_mode=args.gen_mode,
                    use_fp16 = args.use_fp16,
                    class_scale=args.classifier_scale
                )

            # video = ((sample["video"] + 1) * 127.5).clamp(0, 255).to(th.uint8)
            video = sample["video"]
            audio = sample["audio"]              
            # video = video.permute(0, 1, 3, 4, 2)
            # video = video.contiguous()

            all_videos = video.detach().cpu().numpy()
            all_audios = audio.detach().cpu().numpy()

            if args.class_cond:
                all_labels.append(classes.cpu().numpy())
                
            videos.append(all_videos)
            audios.append(all_audios)

            groups += 1

            dist.barrier()

        videos = np.concatenate(videos)
        audios = np.concatenate(audios)
        all_labels = np.concatenate(all_labels) if all_labels != [] else np.zeros(audios.shape[0])
               
        video_output_path = os.path.join(img_save_path, f"RNA_{dist.get_rank()}.npz")
        
        audio_output_path = os.path.join(audio_save_path, f"ATAC_{dist.get_rank()}.npz")
        
        np.savez(video_output_path,data=videos,label=all_labels)
        np.savez(audio_output_path,data=audios,label=all_labels)

        # # calculate metric
        # if os.path.exists(args.ref_path):
        #     for fake_path in [multimodal_save_path, sr_save_path]:
        #         # if fake_path == multimodal_save_path: 
        #         #     video_size = args.video_size
        #         # elif fake_path == sr_save_path: 
        #         #     video_size = [args.video_size[0], args.video_size[1], args.large_size, args.large_size]
                   
        #         metric=eval_multimodal(args.ref_path, multimodal_save_path, eval_num=args.all_save_num)
        #         if dist.get_rank() == 0:
        #             logger.log(f"evaluate between {fake_path} and {args.ref_path}")
        #             logger.log(metric)
        #             delete_pkl(fake_path)


    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        ref_path="",
        batch_size=16,
        sample_fn="dpm_solver",
        sr_sample_fn="dpm_solver",
        sr_model_path="",
        multimodal_model_path="",
        output_dir="",
        save_type="mp4",
        classifier_scale=0.0,
        devices=None,
        is_strict=True,
        all_save_num= 1024,
        seed=42,
        video_fps=10,
        audio_fps=16000,
        load_noise="",
        use_cfgen=False,
        data_dir="",
        gen_mode='atac',
        num_class=33, #22 #14
        class_cond=True,
        encoder_config='encoder_multimodal',
        condition='leiden',
        ae_path='/stor/lep/workspace/multi_diffusion/CFGen/project_folder/experiments/train_autoencoder_babel_multimodal/checkpoints/epoch_39.ckpt',
    )
   
    defaults.update(model_and_diffusion_defaults())
    defaults.update(image_sr_model_and_diffusion_defaults())
    # defaults['num_class'] = 22
    # defaults['class_cond'] = True
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    print(th.cuda.current_device())
    main()
