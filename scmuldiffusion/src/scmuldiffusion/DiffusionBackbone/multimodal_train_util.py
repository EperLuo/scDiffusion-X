import copy
import functools
import os
import blobfile as bf
import torch as th
import torch.distributed as dist
import wandb
import socket
import random
import glob
import numpy as np
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import ExponentialLR
from einops import rearrange, repeat
from . import dist_util, logger
from .fp16_util import MixedPrecisionTrainer
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler
from .multimodal_dpm_solver_plus import DPM_Solver
from .common import save_one_video

INITIAL_LOG_LOSS_SCALE = 20.0
class TrainLoop:
    def __init__(
        self,
        *,
        model,
        diffusion,
        data,
        batch_size,
        microbatch,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        lr=0,
        t_lr=1e-4,
        save_type="mp4",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
        class_cond=False,
        use_db=False,
        sample_fn='dpm_solver',
        num_classes=0,
        save_row=2,
        video_fps=16,
        audio_fps=16000      

    ):
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.save_type = save_type
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.t_lr = t_lr
       
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps
        self.class_cond=class_cond
        self.num_classes=num_classes
        self.save_row = save_row
        self.step = 1
        self.resume_step = 0
        self.global_batch = self.batch_size * dist.get_world_size()
        self.video_fps = video_fps
        self.audio_fps = audio_fps
        self.use_db = use_db
        if self.use_db ==True and dist.get_rank()==0:
            wandb.login(key="<use_your_own_wandb_key>")
            wandb.init(
               project=f"{logger.get_dir().split('/')[-2]}",
               entity="mm-diffusion",
               notes=socket.gethostname(),
               name=f"{logger.get_dir().split('/')[-1]}",
               job_type="training",
               reinit=True)

        self.sync_cuda = th.cuda.is_available()
        self.sample_fn=sample_fn

        self._load_and_sync_parameters()
       
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth
        )
    
        self.opt = AdamW(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )
        # self.scheduler = ExponentialLR(self.opt, gamma=0.9)
        # self.scheduler = ReduceLROnPlateau(self.opt, mode='min', factor=0.5, patience=500)
        
        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.mp_trainer.master_params)
                for _ in range(len(self.ema_rate))
            ]

        self.output_model_stastics()

        
        print("******DDP sync model...")
        if th.cuda.is_available():
            self.use_ddp = True   
            self.ddp_model = DDP(
                self.model,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=True,
            )
            print("******DDP sync model done...")
           
       
        else:
            if dist.get_world_size() > 1:
                logger.warn(
                    "Distributed training requires CUDA. "
                    "Gradients will not be synchronized properly!"
                )
            self.use_ddp = False
            self.ddp_model = self.model
        
    def output_model_stastics(self):
        num_params_total = sum(p.numel() for p in self.model.parameters())
        num_params_train = 0
        num_params_pre_load = 0
        
        for param_group in self.opt.param_groups:
            if param_group['lr'] >0:
                num_params_train += sum(p.numel() for p in param_group['params'] if p.requires_grad==True)
    
        if hasattr(self, 'pre_load_params'):
            num_params_pre_load=sum(p.numel() for name, p in self.model.named_parameters() if name in self.pre_load_params)
            #[p.mean().item() for _, p in self.model.named_parameters()]
        if num_params_total > 1e6:
            num_params_total /= 1e6
            num_params_train /= 1e6
            num_params_pre_load /= 1e6
            params_total_label = 'M'
        elif num_params_total > 1e3:
            num_params_total /= 1e3
            num_params_train /= 1e3
            num_params_pre_load = 1e3
            params_total_label = 'k'

        logger.log("Total Parameters:{:.2f}{}".format(num_params_total, params_total_label))
        logger.log("Total Training Parameters:{:.2f}{}".format(num_params_train, params_total_label))
        logger.log("Total Loaded Parameters:{:.2f}{}".format(num_params_pre_load, params_total_label))
 

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            if self.resume_step > 0 and dist.get_rank()==0:
                logger.log(f"continue training from step {self.resume_step}")
            #if dist.get_rank() == 0:
      
            state_dict = dist_util.load_state_dict(resume_checkpoint, map_location=dist_util.dev())
            self.pre_load_params = state_dict.keys()
            logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
            self.model.load_state_dict_(
                state_dict
                )

        dist_util.sync_params(self.model.parameters()) 
        

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.mp_trainer.master_params)
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)

        if ema_checkpoint:
            logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
            state_dict = dist_util.load_state_dict(
                ema_checkpoint, map_location=dist_util.dev()
            )
            ema_params = self.mp_trainer.state_dict_to_master_params(state_dict)

        dist_util.sync_params(ema_params)
        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)

    def run_loop(self):

        while (
            not self.lr_anneal_steps
            or self.step + self.resume_step < self.lr_anneal_steps
        ):
                    
            batch = next(self.data)
        
            loss = self.run_step(batch,{'label':batch['label']})
           
            if dist.get_rank() == 0 and self.use_db:
                wandb_log = { 'loss': loss["loss"].mean().item()}
                
            if self.step % self.log_interval == 0:
                log = logger.get_current()
                if dist.get_rank() == 0 and self.use_db:
                    wandb_log.update({'grad_norm':log.name2val["grad_norm"], 'loss_q0':log.name2val["loss_q0"], \
                        'v_grad':log.name2val["grad_norm_v"], 'a_grad':log.name2val["grad_norm_a"]})
                logger.dumpkvs()

            if self.step % self.save_interval == 0:
               
                self.save()

            if dist.get_rank() == 0 and self.use_db:
                wandb.log(wandb_log)
            self.step += 1
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self, batch, cond={}):
        self.mp_trainer.zero_grad()  
        loss = self.forward_backward(batch, cond)
        took_step = self.mp_trainer.optimize(self.opt)
        
        if took_step:
            self._update_ema()
        
        self._anneal_lr()
        self.log_step()
        
        return loss

    def forward_backward(self, batch, cond):
        batch = {k:v.to(dist_util.dev()) \
            for k, v in batch.items()}

        cond = {k:v.to(dist_util.dev()) \
            for k, v in cond.items()}

        batch_len = batch['video'].shape[0]
               
        for i in range(0, batch_len, self.microbatch):
            micro = {
                k: v[i : i + self.microbatch]
                for k, v in batch.items()
            }
            
            micro_cond = {
                k: v[i : i + self.microbatch]
                for k, v in cond.items()
            }
            
            last_batch = (i + self.microbatch) >= batch_len
            t, weights = self.schedule_sampler.sample(self.batch_size, dist_util.dev())
            
           
            compute_losses = functools.partial(
            self.diffusion.multimodal_training_losses,
            self.ddp_model,
            micro,
            t,
            model_kwargs=micro_cond,
            )
          
            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            loss = (losses["loss"] * weights).mean()
            self.mp_trainer.backward(loss)
         
        if isinstance(self.schedule_sampler, LossAwareSampler):
            self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

        log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
           
        return losses

    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.mp_trainer.master_params, rate=rate)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)
    
    def save(self):
        def save_checkpoint(rate, params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                logger.log(f"saving model {rate}...")
                if not rate:
                    filename = f"model{(self.step+self.resume_step):06d}.pt"
                else:
                    filename = f"ema_{rate}_{(self.step+self.resume_step):06d}.pt"
                with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                    th.save(state_dict, f)

        save_checkpoint(0, self.mp_trainer.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        if dist.get_rank() == 0:
            with bf.BlobFile(
                bf.join(get_blob_logdir(), f"opt{(self.step+self.resume_step):06d}.pt"),
                "wb",
            ) as f:
                th.save(self.opt.state_dict(), f)

        dist.barrier()


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    filename = "model*.pt"
    max_step = 0
    for name in glob.glob(os.path.join(get_blob_logdir(), filename)):
        step = int(name[-9:-3])
        max_step = max(max_step, step)
    if max_step:
        path = bf.join(get_blob_logdir(), f"model{(max_step):06d}.pt")
    
        if bf.exists(path):
            return path
    return None

def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
