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

from scdiffusionX.DiffusionBackbone import dist_util, logger
from scdiffusionX.DiffusionBackbone.multimodal_script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict
)
from scdiffusionX.DiffusionBackbone.common import set_seed_logger_random, delete_pkl
from scdiffusionX.DiffusionBackbone.multimodal_dpm_solver_plus import DPM_Solver as multimodal_DPM_Solver
from scdiffusionX.DiffusionBackbone.dpm_solver_plus import DPM_Solver as singlemodal_DPM_Solver

def main():
    args = create_argparser().parse_args()
    args.rna_dim = [int(i) for i in args.rna_dim.split(',')]
    args.atac_dim = [int(i) for i in args.atac_dim.split(',')]
    
    
    dist_util.setup_dist(args.devices)
    logger.configure(args.output_dir)
    args = set_seed_logger_random(args)


    logger.log("creating model and diffusion...")
    multimodal_model, multimodal_diffusion = create_model_and_diffusion(
         **args_to_dict(args, [key for key in model_and_diffusion_defaults().keys()])
    )

    if os.path.isdir(args.multimodal_model_path):
        multimodal_name_list = [model_name for model_name in os.listdir(args.multimodal_model_path) \
            if (model_name.startswith('model') and model_name.endswith('.pt') and int(model_name.split('.')[0][5:])>= args.skip_steps)]
        multimodal_name_list.sort()
        multimodal_name_list = [os.path.join(args.model_path, model_name) for model_name in multimodal_name_list[::1]]
    else:
        multimodal_name_list = [model_path for model_path in args.multimodal_model_path.split(',')]
        
    logger.log(f"models waiting to be evaluated:{multimodal_name_list}")

    sr_noise=None
    if os.path.exists(args.load_noise):
        sr_noise = np.load(args.load_noise)
        sr_noise = th.tensor(sr_noise).to(dist_util.dev()).unsqueeze(0)
        sr_noise = repeat(sr_noise, 'b c h w -> (b repeat) c h w', repeat=args.batch_size * args.rna_dim[0])
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
        multimodal_model.specific_type = 0

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

        if args.class_cond:
            mdata = mu.read_h5mu(args.data_dir)
            from sklearn.preprocessing import LabelEncoder
            labels = mdata['rna'].obs[args.condition].values
            label_encoder = LabelEncoder()
            label_encoder.fit(labels)
            classes_all = label_encoder.transform(labels)
            # classes_all = classes_all[np.where(np.in1d(classes_all,[20,])==1)[0]]

        videos = []
        audios = []
        all_labels = []

        while groups * args.batch_size *  dist.get_world_size()< args.all_save_num: 
            model_kwargs = {}

            if args.class_cond:
                if args.specific_type is None:
                    classes = np.random.choice(classes_all, args.batch_size, replace=False)  # generated random cell type
                else:
                    print(f'generating {label_encoder.classes_[int(args.specific_type)]} cell')
                    classes = th.ones(args.batch_size)*int(args.specific_type)   # generated certain cell type
                classes = th.tensor(classes, device=dist_util.dev(), dtype=th.int)
                model_kwargs["label"] = classes

            shape = {"video":(args.batch_size , *args.rna_dim), \
                    "audio":(args.batch_size , *args.atac_dim)
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
                    multimodal_diffusion.p_sample_loop if  args.sample_fn=="ddpm" else multimodal_diffusion.ddim_sample_loop
                )

                sample = sample_fn(
                    multimodal_model,
                    shape = shape,
                    clip_denoised=args.clip_denoised,
                    model_kwargs=model_kwargs,
                )

            video = sample["video"]
            audio = sample["audio"]              

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

    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        ref_path="",
        batch_size=16,
        sample_fn="dpm_solver",
        multimodal_model_path="",
        output_dir="",
        classifier_scale=0,
        devices=None,
        is_strict=True,
        all_save_num= 1024,
        seed=42,
        load_noise="",
        data_dir="",
        condition='cell_type',
        specific_type=None,
    )
   
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    print(th.cuda.current_device())
    main()
