from abc import abstractmethod
import random
import math
import time
from einops import rearrange
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from . import logger
from .fp16_util import convert_module_to_f16, convert_module_to_f32
from .nn import (
    checkpoint,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    normalization_cell,
    timestep_embedding
)
import numpy as np
import os


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, video, audio, emb):#
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, video, audio, emb):#
        for layer in self:
            if isinstance(layer, TimestepBlock):
                video, audio = layer(video, audio, emb)
            else:
                if isinstance(layer, CrossAttentionBlock_cell):
                    video, audio, rna_vec, atac_vec, att_vec1, att_vec2 = layer(video, audio)
                    return video, audio, rna_vec, atac_vec, att_vec1, att_vec2
                else:
                    video, audio = layer(video, audio)
        return video, audio
        


class InitialTransfer(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb=None, temporal_emb=None):#
        b, f, c,h, w = x.shape
        x = rearrange(x, 'b f c h w -> (b f) c h w')
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        
        x = rearrange(x, '(b f) c h w -> b f c h w' , b=b, f=f)
        return x

class VideoConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride = 1,
        padding = "same",
        dilation = 1,
        conv_type = '2d+1d'
    ):
        super().__init__()
        self.conv_type = conv_type
        self.padding= padding

        if conv_type == '2d+1d':
            self.video_conv_spatial = conv_nd(2, in_channels, out_channels, kernel_size, stride, padding, dilation)
            self.video_conv_temporal = conv_nd(1, out_channels,out_channels, kernel_size, stride, padding, dilation)
        elif conv_type == '3d':
            self.video_conv = conv_nd(3, in_channels, out_channels,kernel_size, stride=stride, padding=padding, dilation=dilation)
        else:
            raise NotImplementedError

    def forward(self, video):
        if self.conv_type == '2d+1d':
            
            b, f, c, h, w = video.shape
            video = rearrange(video, 'b f c h w -> (b f) c h w')
            video = self.video_conv_spatial(video)
            video = rearrange(video, '(b f) c h w -> (b h w) c f', b=b)
            video = self.video_conv_temporal(video)
            video = rearrange(video, '(b h w) c f -> b f c h w', b=b, h=h)

        elif self.conv_type == '3d':
            video = rearrange(video, 'b f c h w -> b c f h w')
            video = self.video_conv(video)
            video = rearrange(video, 'b c f h w -> b f c h w')

        return video


class CellConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        feature_dim,
        kernel_size=3,
        stride = 1,
        padding = "same",
        dilation = 1,
        conv_type = '1d',
    ):
        super().__init__()
        
        # if conv_type == '1d':
        #     self.cell_conv = conv_nd(1, in_channels, out_channels, kernel_size, stride, padding, dilation)
         
        # elif conv_type == 'linear':
        #     self.cell_conv = conv_nd(1, in_channels, out_channels, kernel_size, stride, padding, dilation)
        # else:
        #     raise NotImplementedError
        self.cell_conv = nn.Linear(in_channels, out_channels)
        self.out_channels = out_channels
        # if out_channels!=1:
        #     self.selfattension = nn.MultiheadAttention(embed_dim=out_channels, num_heads=8)

        #     self.q = nn.Linear(out_channels,out_channels)
        #     self.k = nn.Linear(out_channels,out_channels)
        #     self.v = nn.Linear(out_channels,out_channels)
            
        #     self.linear = nn.Linear(out_channels,out_channels)
        
    def forward(self, audio):
        audio = self.cell_conv(audio)

        # if self.out_channels!=1:
        #     q = self.q(audio)
        #     k = self.k(audio)
        #     v = self.v(audio)
        #     audio, attn_output_weights = self.selfattension(q, k, v, attn_mask=None)

        #     audio = self.linear(audio)
        return audio#.permute(0,2,1)

class AudioConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride = 1,
        padding = "same",
        dilation = 1,
        conv_type = '1d',
    ):
        super().__init__()
        
        if conv_type == '1d':
            self.audio_conv = conv_nd(1, in_channels, out_channels, kernel_size, stride, padding, dilation)
         
        elif conv_type == 'linear':
            self.audio_conv = conv_nd(1, in_channels, out_channels, kernel_size, stride, padding, dilation)
        else:
            raise NotImplementedError
        
    def forward(self, audio):
        audio = self.audio_conv(audio)
        return audio

class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims=dims
        if dims == 3: 
            # for video
            self.stride = (1,2,2)
        elif dims == 1:
            #for audio
            self.stride = 2
        else:
            # for image
            self.stride = 2

        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 2, padding='same')

    def forward(self, x):
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2] * self.stride[0], x.shape[3] * self.stride[1], x.shape[4] * self.stride[2]), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=self.stride, mode="nearest")

        if self.use_conv:
            x = self.conv(x)
   
        return x

class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if dims == 3:
            stride = (1,2,2)
        elif dims == 1:
            stride = 2
        else:
            stride = 2

        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 2, stride=stride
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        x = self.op(x)
        return x



class SingleModalQKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)

class SingleModalAtten(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        feature_dim=None,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        if feature_dim is not None:
            self.norm = normalization_cell(channels)
        else:
            self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        self.attention = SingleModalQKVAttention(self.num_heads)
        
        self.proj_out = conv_nd(1, channels, channels, 1)

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), True)

    def _forward(self, x):
        
        b, c, *spatial = x.shape
     
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return x + h.reshape(b, c, *spatial)



class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.

    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param video_type: the type of video model to use.
    :param audio_type: the type of audio model to use.
    :param audio_dilation: the dilation to use for the audio convolution.
    :param video_attention: if True, use attention in the video model.
    :param audio_attention: if True, use attention in the audio model.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        video_type='2d+1d',
        audio_type='1d',
        audio_dilation=1,
        use_scale_shift_norm=False,
        use_checkpoint=False,
        up=False,
        down=False,
        use_conv=False,
        video_attention = False,
        audio_attention = False,
        num_heads=4,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm
        self.video_in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            VideoConv(channels, self.out_channels, 3,  conv_type=video_type),
        )
        self.audio_in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            AudioConv(channels, self.out_channels, 3,  conv_type=audio_type,  dilation=audio_dilation),
        )
        
        self.updown = up or down
        self.video_attention = video_attention
        self.audio_attention = audio_attention

        if up:
            self.vh_upd = Upsample(channels, False, 3)
            self.vx_upd = Upsample(channels, False, 3)
            self.ah_upd = Upsample(channels, False, 1)
            self.ax_upd = Upsample(channels, False, 1)
        elif down:
            self.vh_upd = Downsample(channels, False, 3)
            self.vx_upd = Downsample(channels, False, 3)
            self.ah_upd = Downsample(channels, False, 1)
            self.ax_upd = Downsample(channels, False, 1)
        else:
            self.ah_upd = self.ax_upd = self.vh_upd = self.vx_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.video_out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                VideoConv(self.out_channels, self.out_channels, 1,  conv_type='3d')
            ),
        )
        self.audio_out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                AudioConv(self.out_channels, self.out_channels, 1, conv_type='linear')
            ),
        )

        if self.out_channels == channels:
            self.video_skip_connection = nn.Identity()
            self.audio_skip_connection = nn.Identity()
        elif use_conv:
            self.video_skip_connection = VideoConv(
                 channels, self.out_channels, 3, conv_type='2d+1d'
            )
            self.audio_skip_connection = AudioConv(
                 channels, self.out_channels, 3, conv_type='1d'
            )
        else:
            self.video_skip_connection = VideoConv(
                 channels, self.out_channels, 1, conv_type='3d'
            )
            self.audio_skip_connection = AudioConv(
                 channels, self.out_channels, 1, conv_type='linear'
            )

        
        if self.video_attention:
            self.spatial_attention_block = SingleModalAtten(
                channels=self.out_channels, num_heads=num_heads, 
                num_head_channels=-1, use_checkpoint=use_checkpoint)
            self.temporal_attention_block = SingleModalAtten(
                channels=self.out_channels, num_heads=num_heads, 
                num_head_channels=-1, use_checkpoint=use_checkpoint)
        if self.audio_attention:
            self.audio_attention_block = SingleModalAtten(
                channels=self.out_channels, num_heads=num_heads, 
                num_head_channels=-1, use_checkpoint=use_checkpoint)
       

    def forward(self, video, audio, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (video, audio, emb), self.parameters(), self.use_checkpoint
        )

    def _forward(self, video, audio, emb):
        '''
        video:(b,f,c,h,w)
        audio:(b,c,l)
        emb:(b,c)
        '''
        b, f, c, h, w = video.shape
        if self.updown:
            video_h = self.video_in_layers(video)
            video_h = self.vh_upd(video_h)
            video = self.vx_upd(video)

            audio_h = self.audio_in_layers(audio)
            audio_h = self.ah_upd(audio_h)
            audio = self.ax_upd(audio)

        else:
            video_h = self.video_in_layers(video)
            audio_h = self.audio_in_layers(audio)
            
        emb_out = self.emb_layers(emb).type(video.dtype)
        

        if self.use_scale_shift_norm:
            # use the same embed
            video_out_norm, video_out_rest = self.video_out_layers[0], self.video_out_layers[1:]
            video_emb_out = emb_out[:,None,:,None,None]
          
            scale, shift = th.chunk(video_emb_out, 2, dim=2)
            video_h = video_out_norm(video_h) * (1 + scale) + shift
            video_h = video_out_rest(video_h)

            audio_out_norm, audio_out_rest = self.audio_out_layers[0], self.audio_out_layers[1:]
            audio_emb_out = emb_out[...,None]
            scale, shift = th.chunk(audio_emb_out, 2, dim=1)
            audio_h = audio_out_norm(audio_h) * (1 + scale) + shift
            audio_h = audio_out_rest(audio_h)

        else:
            video_emb_out = emb_out[:,None,:,None,None]
            video_h = video_h + video_emb_out
            video_h = self.video_out_layers(video_h)
            audio_emb_out = emb_out[...,None]
            audio_h = audio_h + audio_emb_out
            audio_h = self.audio_out_layers(audio_h)

        

        video_out = self.video_skip_connection(video) + video_h
        audio_out = self.audio_skip_connection(audio) + audio_h

        if self.video_attention:
           
            video_out= rearrange(video_out, "b f c h w -> (b f) c (h w)")
            video_out = self.spatial_attention_block(video_out)
            video_out= rearrange(video_out, "(b f) c (h w) -> (b h w) c f", f=f, h=h)
            video_out = self.temporal_attention_block(video_out)
            video_out= rearrange(video_out, "(b h w) c f -> b f c h w", h=h, w=w)
        if self.audio_attention:
            audio_out = self.audio_attention_block(audio_out)

        return video_out, audio_out

class ResBlock_cell(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.

    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param video_type: the type of video model to use.
    :param audio_type: the type of audio model to use.
    :param audio_dilation: the dilation to use for the audio convolution.
    :param video_attention: if True, use attention in the video model.
    :param audio_attention: if True, use attention in the audio model.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_scale_shift_norm=False,
        use_checkpoint=False,
        up=False,
        down=False,
        use_conv=False,
        num_heads=4,
        feature_dim=26,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm
        self.video_in_layers = nn.Sequential(
            normalization_cell(channels),
            nn.SiLU(),
            CellConv(channels, self.out_channels, feature_dim, kernel_size=3,  conv_type='1d'),
        )
        self.audio_in_layers = nn.Sequential(
            normalization_cell(channels),
            nn.SiLU(),
            CellConv(channels, self.out_channels, feature_dim, kernel_size=3,  conv_type='1d'),
        )
        
        self.updown = up or down
        self.video_attention = False #video_attention
        self.audio_attention = False #audio_attention

        if up:
            self.vh_upd = Upsample(channels, True, 1)
            self.vx_upd = Upsample(channels, True, 1)
            self.ah_upd = Upsample(channels, True, 1)
            self.ax_upd = Upsample(channels, True, 1)
            feature_dim = int(feature_dim*2)
        elif down:
            self.vh_upd = Downsample(channels, True, 1)
            self.vx_upd = Downsample(channels, True, 1)
            self.ah_upd = Downsample(channels, True, 1)
            self.ax_upd = Downsample(channels, True, 1)
            feature_dim = int(feature_dim/2)
        else:
            self.ah_upd = self.ax_upd = self.vh_upd = self.vx_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )

        self.video_out_layers = nn.Sequential(
            normalization_cell(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            CellConv(self.out_channels, self.out_channels, feature_dim, kernel_size=1,  conv_type='1d'),
        )
        self.audio_out_layers = nn.Sequential(
            normalization_cell(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            CellConv(self.out_channels, self.out_channels, feature_dim, kernel_size=1, conv_type='1d'),
        )

        if self.out_channels == channels:
            self.video_skip_connection = nn.Identity()
            self.audio_skip_connection = nn.Identity()
        elif use_conv:
            self.video_skip_connection = CellConv(
                 channels, self.out_channels, feature_dim, kernel_size=3, conv_type='1d'
            )
            self.audio_skip_connection = CellConv(
                 channels, self.out_channels, feature_dim, kernel_size=3, conv_type='1d'
            )
        else:
            self.video_skip_connection = CellConv(
                 channels, self.out_channels, feature_dim, kernel_size=1, conv_type='1d'
            )
            self.audio_skip_connection = CellConv(
                 channels, self.out_channels, feature_dim, kernel_size=1, conv_type='1d'
            )

        
        if self.video_attention:
            self.video_attention_block = SingleModalAtten(
                channels=self.out_channels, num_heads=num_heads, 
                num_head_channels=-1, use_checkpoint=use_checkpoint,feature_dim=feature_dim)
        if self.audio_attention:
            self.audio_attention_block = SingleModalAtten(
                channels=self.out_channels, num_heads=num_heads, 
                num_head_channels=-1, use_checkpoint=use_checkpoint,feature_dim=feature_dim)
       

    def forward(self, video, audio, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (video, audio, emb), self.parameters(), self.use_checkpoint
        )

    def _forward(self, video, audio, emb):
        '''
        video:(b,f,c,h,w)
        audio:(b,c,l)
        emb:(b,c)
        '''
        if self.updown:
            video_h = self.video_in_layers(video)
            video_h = self.vh_upd(video_h)
            video = self.vx_upd(video)

            audio_h = self.audio_in_layers(audio)
            audio_h = self.ah_upd(audio_h)
            audio = self.ax_upd(audio)

        else:
            video_h = self.video_in_layers(video)
            audio_h = self.audio_in_layers(audio)
            
        emb_out = self.emb_layers(emb).type(video.dtype)
        

        if self.use_scale_shift_norm:
            # use the same embed
            video_out_norm, video_out_rest = self.video_out_layers[0], self.video_out_layers[1:]
            video_emb_out = emb_out[:,None,:]
            scale, shift = th.chunk(video_emb_out, 2, dim=-1)
            video_h = video_out_norm(video_h) * (1 + scale) + shift
            video_h = video_out_rest(video_h)

            audio_out_norm, audio_out_rest = self.audio_out_layers[0], self.audio_out_layers[1:]
            audio_emb_out = emb_out[:,None,:]
            scale, shift = th.chunk(audio_emb_out, 2, dim=-1)
            audio_h = audio_out_norm(audio_h) * (1 + scale) + shift
            audio_h = audio_out_rest(audio_h)

        else:
            video_emb_out = emb_out[:,None,:]
            video_h = video_h + video_emb_out
            video_h = self.video_out_layers(video_h)
            audio_emb_out = emb_out[:,None,:]
            audio_h = audio_h + audio_emb_out
            audio_h = self.audio_out_layers(audio_h)

        

        video_out = self.video_skip_connection(video) + video_h
        audio_out = self.audio_skip_connection(audio) + audio_h

        if self.video_attention:
            video_out = self.video_attention_block(video_out)
        if self.audio_attention:
            audio_out = self.audio_attention_block(audio_out)

        return video_out, audio_out


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv, video_attention_index, audio_attention_index, frame_size, audio_per_frame):
        """
        Apply QKV attention.
        : attention_index_v:[V_len x H]
        : attention_index_a:[A_len, H]
        :param qkv: an [ N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
    
        
        bs, width, _ = qkv.shape
        video_len = video_attention_index.shape[0] 
        audio_len = audio_attention_index.shape[0]
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))

        v_as = []
        a_as = []
        video_q = q[:, :, :video_len] #[bsz, c*head, videolength]
        audio_q = q[:, :, video_len:] 
      
        for idx in range(0, video_len//frame_size):
            video_frame_k = th.index_select(k, -1, video_attention_index[idx*frame_size]) #[bsz, c*head, k_num]
            video_frame_v = th.index_select(v, -1, video_attention_index[idx*frame_size]) #[bsz, c*head, k_num]
            video_frame_q = video_q[:, :, idx*frame_size:(idx+1) * frame_size]
            
            w_slice =  th.einsum(
            "bct,bcs->bts",
            (video_frame_q * scale).view(bs * self.n_heads, ch, -1),
            (video_frame_k * scale).view(bs * self.n_heads, ch, -1),
            )  # More stable with f16 than dividing afterwards
            
            w_slice = th.softmax(w_slice, dim=-1) #[bsz, 1, k_len]
            a = th.einsum("bts,bcs->bct", w_slice, video_frame_v.view(bs * self.n_heads, ch, -1)).reshape(bs * self.n_heads, ch, -1)
            v_as.append(a)

            audio_frame_k = th.index_select(k, -1, audio_attention_index[idx*audio_per_frame])#[bsz, c*head, k_num]
            audio_frame_v = th.index_select(v, -1, audio_attention_index[idx*audio_per_frame]) #[bsz, c*head, k_num]
            if idx == (video_len//frame_size-1): 
                audio_frame_q = audio_q[:, :, idx*audio_per_frame:]
            else:
                audio_frame_q = audio_q[:, :, idx*audio_per_frame:(idx +1)* audio_per_frame]
            w_slice =  th.einsum(
            "bct,bcs->bts",
            (audio_frame_q * scale).view(bs * self.n_heads, ch, -1),
            (audio_frame_k * scale).view(bs * self.n_heads, ch, -1),
            )  # More stable with f16 than dividing afterwards

            w_slice = th.softmax(w_slice, dim=-1) #[bsz, 1, k_len]
            a = th.einsum("bts,bcs->bct", w_slice, audio_frame_v.view(bs * self.n_heads, ch, -1)).reshape(bs * self.n_heads, ch, -1)
            a_as.append(a)
      
        v_a = th.cat(v_as, dim=2)
        a_a = th.cat(a_as, dim=2)

        return v_a.reshape(bs, -1,  video_len), a_a.reshape(bs, -1, audio_len)
    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class CrossAttentionBlock(nn.Module):
    """
    RS-MMA: ramdom based multi-modal attention block
    An attention block that allows cross attention .
    :param local_window: local window size.
    :param window_shift: whether to random shift the window.
    """

    def __init__(
        self,
        channels,
  
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        local_window = 1,
        window_shift = False,
    ):
        super().__init__()
        self.channels = channels

        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                self.channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            
            self.num_heads = self.channels // num_head_channels

        self.local_window = local_window
        self.window_shift = window_shift
        self.use_checkpoint = use_checkpoint
        self.v_norm = normalization(self.channels)
        self.a_norm = normalization(self.channels)
        self.v_qkv = conv_nd(1, self.channels, self.channels * 3, 1)
        self.a_qkv = conv_nd(1, self.channels, self.channels * 3, 1)
        self.attention = QKVAttention(self.num_heads)

        self.video_proj_out = zero_module(VideoConv(self.channels, self.channels, 1, conv_type = '3d'))
        self.audio_proj_out = zero_module(AudioConv(self.channels, self.channels, 1, conv_type = 'linear'))
        self.va_index=None
        self.av_index=None
 
    def attention_index(self, atac_dim, rna_dim, device):
        f, h, w = rna_dim
        l = atac_dim
        video_len = f * h * w
        audio_len_perf = int(l / f)
        if self.window_shift:
            window_shift = random.randint(0, f-self.local_window)
        else:
            window_shift = 0
        
        if self.va_index == None:
            va_index_x =  th.arange(0,  self.local_window*audio_len_perf).view(1, -1) 
            va_index_y = th.arange(0, f).unsqueeze(-1).repeat(1, h*w).view(-1, 1)
            va_index_y = va_index_y * audio_len_perf
            self.va_index = (va_index_y + va_index_x).to(device)

        va_index = (self.va_index +audio_len_perf*window_shift) %l + video_len

        if self.av_index == None:
            av_index_x = th.arange(0,  self.local_window*h*w).view(1, -1)
            av_index_y = th.arange(0, f).unsqueeze(-1).repeat(1, audio_len_perf).view(-1, 1)
            av_index_y = av_index_y * h*w
            self.av_index =  (av_index_y + av_index_x).to(device)

        av_index = (self.av_index + h*w*window_shift)%video_len
        
        attention_index_v = va_index
        attention_index_a = av_index
        
        # complete attention index
        if attention_index_a.shape[0] < l:
            attention_index_a = th.cat([attention_index_a, attention_index_a[-1*(l-attention_index_a.shape[0]):]], dim=0)
    
        return attention_index_v, attention_index_a


    def forward(self, video, audio): 
        
        return checkpoint(self._forward, (video, audio), self.parameters(), True)
        

    def _forward(self, video, audio): 
        b, f, c, h, w = video.shape
        b, c, l = audio.shape
        
        video_token = rearrange(video, 'b f c h w -> b c (f h w)')
        audio_token = audio

        attention_index_v, attention_index_a = self.attention_index((l), (f, h, w), video.device)

        v_qkv = self.v_qkv(self.v_norm(video_token)) #[bsz, c, f*h*w+l]
        a_qkv = self.a_qkv(self.a_norm(audio_token)) #[bsz, c, f*h*w+l]
        qkv = th.concat([v_qkv, a_qkv], dim=2)

        video_h, audio_h = self.attention(qkv, attention_index_v, attention_index_a, h*w, int(l/f))

        video_h = rearrange(video_h, 'b c (f h w)-> b f c h w ', f=f, h=h)
        video_h = self.video_proj_out(video_h)
        video_h = video + video_h
 
        audio_h = self.audio_proj_out(audio_h)
        audio_h = audio + audio_h
      
        
        return video_h, audio_h

class CrossAttentionBlock_cell(nn.Module):
    """
    RS-MMA: ramdom based multi-modal attention block
    An attention block that allows cross attention .
    :param local_window: local window size.
    :param window_shift: whether to random shift the window.
    """

    def __init__(
        self,
        channels,
  
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        local_window = 1,
        window_shift = False,
        feature_dim=24,
        id = 1,
        specific_type=None,
    ):
        super().__init__()
        self.channels = channels
        self.feature_dim = 64
        self.id = id
        self.time_step = 0
        self.specific_type=specific_type

        # if num_head_channels == -1:
        #     self.num_heads = num_heads
        # else:
        #     assert (
        #         self.feature_dim % num_head_channels == 0
        #     ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            
        #     self.num_heads = self.feature_dim // num_head_channels

        # self.local_window = local_window
        # self.window_shift = window_shift
        # self.use_checkpoint = use_checkpoint
        # self.v_norm = normalization_cell(feature_dim)
        # self.k_norm = normalization_cell(feature_dim)
        # self.q_norm = normalization_cell(feature_dim)
        # # self.v_qkv = conv_nd(1, self.channels, self.channels * 3, 1)
        # # self.a_qkv = conv_nd(1, self.channels, self.channels * 3, 1)

        # if cross attention

        self.q_rna = nn.Linear(1,self.feature_dim,bias=False)
        self.k_rna = nn.Linear(1,self.feature_dim,bias=False)
        self.v_rna = nn.Linear(1,self.feature_dim,bias=False)

        self.q_atac = nn.Linear(1,self.feature_dim,bias=False)
        self.k_atac = nn.Linear(1,self.feature_dim,bias=False)
        self.v_atac = nn.Linear(1,self.feature_dim,bias=False)

        self.trans_rna = nn.Linear(self.feature_dim,1,bias=False)
        self.trans_atac = nn.Linear(self.feature_dim,1,bias=False)

        # # if cat linear

        # self.cat_linear = nn.Linear(channels*2,channels*2)
        # self.rna_norm = nn.LayerNorm(channels)
        # self.atac_norm = nn.LayerNorm(channels)


    def forward(self, video, audio): 
        
        return checkpoint(self._forward, (video, audio), self.parameters(), True)
        

    def _forward(self, video, audio): 

        # if cross attention

        video_old1 = video
        audio_old1 = audio

        video = video.transpose(-1,-2)
        audio = audio.transpose(-1,-2)

        q_rna = self.q_rna(video)
        k_rna = self.k_rna(video)
        v_rna = self.v_rna(video)

        q_atac = self.q_atac(audio)
        k_atac = self.k_atac(audio)
        v_atac = self.v_atac(audio)

        att_rna_atac = F.softmax(th.matmul(q_rna, k_atac.transpose(-1,-2))/th.tensor(self.feature_dim).sqrt().to(device=q_rna.device),dim=-1)
        rna_new = th.matmul(att_rna_atac,v_atac)     # RNA borrow information from ATAC
        att_atac_rna = F.softmax(th.matmul(q_atac, k_rna.transpose(-1,-2))/th.tensor(self.feature_dim).sqrt().to(device=q_atac.device),dim=-1)
        atac_new = th.matmul(att_atac_rna,v_rna)     # ATAC borrow information from RNA
        # video_old1 = k_rna
        # audio_old1 = k_atac
        video_old2 = att_rna_atac
        audio_old2 = att_atac_rna

        # if self.specific_type is not None:
        #     self.time_step += 1
        #     folder_path = f'/stor/lep/workspace/multi_diffusion/MM-Diffusion/outputs/attention_map_open_all/condition_type{self.specific_type}_open'
        #     os.makedirs(folder_path,exist_ok=True)
        #     if self.time_step%100==0 or self.time_step==925 or self.time_step==950 or self.time_step==975:
        #         np.savez(folder_path+f'/att_map_rna_{self.id}_step{self.time_step}.npz',data=att_rna_atac.mean(0).detach().cpu().numpy())
        #         np.savez(folder_path+f'/att_map_atac_{self.id}_step{self.time_step}.npz',data=att_atac_rna.mean(0).detach().cpu().numpy())
        #     if self.time_step==950 or self.time_step==1000 or self.time_step==900 or self.time_step==800:
        #         np.savez(folder_path+f'/att_map_rna_{self.id}_step{self.time_step}_all.npz',data=att_rna_atac.detach().cpu().numpy())
        #         np.savez(folder_path+f'/att_map_atac_{self.id}_step{self.time_step}_all.npz',data=att_atac_rna.detach().cpu().numpy())
        #     if self.time_step == 1000:
        #         self.time_step = 0

        video = video + self.trans_rna(rna_new)
        audio = audio + self.trans_atac(atac_new)

        video = video.transpose(-1,-2)
        audio = audio.transpose(-1,-2)

        # if cat linear
        
        # joint_emb = self.cat_linear(th.concat((video,audio),dim=-1))
        # video_h, audio_h = joint_emb.chunk(2,dim=-1)
        # video_h = self.atac_norm(video_h)
        # audio_h = self.rna_norm(audio_h)
        # audio = audio_h + audio
        # video = video_h + video
        # video_old1 = video
        # audio_old1 = audio
        # video_old2 = video
        # audio_old2 = audio        

        return video, audio, video_old1, audio_old1, video_old2, audio_old2

class InitialBlock(nn.Module):
    def __init__(
        self,
        video_in_channels,
        audio_in_channels,
        video_out_channels,
        audio_out_channels,
        kernel_size = 3 
    ):
        super().__init__()
        self.video_conv = VideoConv(video_in_channels, video_out_channels, kernel_size, conv_type='2d+1d')
        self.audio_conv = AudioConv(audio_in_channels, audio_out_channels, kernel_size, conv_type='linear')

    def forward(self, video, audio): 
        return self.video_conv(video), self.audio_conv(audio)


class InitialBlock_cell(nn.Module):
    def __init__(
        self,
        video_in_channels,
        audio_in_channels,
        video_out_channels,
        audio_out_channels,
        feature_dim,
        kernel_size = 3 
    ):
        super().__init__()
        self.video_conv = CellConv(video_in_channels, video_out_channels,feature_dim=feature_dim, kernel_size=kernel_size, conv_type='1d')
        self.audio_conv = CellConv(audio_in_channels, audio_out_channels,feature_dim=feature_dim, kernel_size=kernel_size, conv_type='1d')

    def forward(self, video, audio): 
        return self.video_conv(video), self.audio_conv(audio)

class MultimodalUNet(nn.Module):
    """
    The full coupled-UNet model with attention and timestep embedding.

    :param rna_dim: the size of the video input.
    :param atac_dim: the size of the audio input.
    :param model_channels: base channel count for the model.
    :param video_out_channels: channels in the output video.
    :param audio_out_channels: channels in the output audio.
    :param num_res_blocks: number of residual blocks per downsample.
    :cross_attention_resolutions: a collection of downsample rates at which cross
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, cross attention
        will be used.
    :cross_attention_windows: a collection of cross-attention window sizes, corressponding 
        to the cross_attention_resolutions.
    :cross_attention_shift: bool. If True, the cross attention window will be shifted randomly
    :param video_attention_resolutions: a collection of downsample rates at which
        video attention will take place. 
    :param audio_attention_resolutions: a collection of downsample rates at which
        audio attention will take place, default -1, which means no audio attention. 
    :param video_type: the layer type for the video encoder, default is '2d+1d', 
        which means 2d conv + 1d conv.
    :param audio_type: the layer type for the audio encoder, default is '1d'.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes, 
        we didn't support class-conditional training.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.

    """

    def __init__(
        self,
        rna_dim,
        atac_dim,
        model_channels,
       
        video_out_channels,
        audio_out_channels,
        num_res_blocks,
        cross_attention_resolutions,
        cross_attention_windows,
        cross_attention_shift,

        dropout=0,
        channel_mult=(1, 2, 3, 4),
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=True,
        specific_type=None,
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.rna_dim = rna_dim.copy()
        self.atac_dim = atac_dim.copy()
        self.specific_type = specific_type

        self.model_channels = model_channels
        self.video_out_channels = video_out_channels
        self.audio_out_channels = audio_out_channels
        self.num_res_blocks = num_res_blocks
        self.cross_attention_resolutions = cross_attention_resolutions
        self.cross_attention_windows = cross_attention_windows
        self.cross_attention_shift = cross_attention_shift
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.num_classes = int(num_classes) if num_classes is not None else num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample

        time_embed_dim = model_channels
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(self.num_classes, time_embed_dim)
        ch = input_ch = int(channel_mult[0] * model_channels)
        
        self._feature_size = ch
        input_block_chans = [ch]
        feature_dim = self.atac_dim[-1]
        self.input_blocks = nn.ModuleList([TimestepEmbedSequential(InitialBlock_cell( self.rna_dim[-1],  self.atac_dim[-1], feature_dim=feature_dim, video_out_channels = ch, audio_out_channels=ch))])

        len_audio_conv = 1

        bid = 1
        dilation = 1
       
        for level, mult in enumerate(channel_mult):
            for block_id in range(num_res_blocks):
                layers=[ResBlock_cell(
                        ch,
                        time_embed_dim,
                        dropout,
                        feature_dim=feature_dim,
                        out_channels=int(mult * model_channels),
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        num_heads = num_heads,
                    )]
                
                dilation += len_audio_conv
                ch = int(mult * model_channels)

                if level == len(channel_mult)-2:              
                    layers.append(CrossAttentionBlock_cell(
                            ch, 
                            feature_dim=feature_dim,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            local_window=self.cross_attention_windows[0],
                            window_shift=self.cross_attention_shift,
                            num_head_channels=num_head_channels,  
                            id=1, 
                            specific_type=self.specific_type,
                            )
                    )

                self.input_blocks.append(TimestepEmbedSequential(*layers))
                bid += 1
                self._feature_size += ch
                input_block_chans.append(ch)
            
        self.middle_blocks = TimestepEmbedSequential(
            CrossAttentionBlock_cell(
                    ch,
                    feature_dim=feature_dim,
                    use_checkpoint=use_checkpoint,
                    num_heads=num_heads,
                    num_head_channels=num_head_channels,
                    local_window = self.rna_dim[0],
                    window_shift = False,
                    id=2,
                    specific_type=self.specific_type,
                ),
            ResBlock_cell(
                    ch,
                    time_embed_dim,
                    dropout,
                    feature_dim=feature_dim,
                    use_checkpoint=use_checkpoint,   
                    use_scale_shift_norm=use_scale_shift_norm,
                    num_heads=num_heads,
            )
        )           
       
        self._feature_size += ch
        bid=0
        self.output_blocks = nn.ModuleList([])
        dilation -= len_audio_conv

        for level, mult in list(enumerate(channel_mult))[::-1]:
            for block_id in range(num_res_blocks):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock_cell(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        feature_dim=feature_dim,
                        out_channels=int(model_channels * mult),
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        num_heads=num_heads 
                    )
                ]

                dilation -= len_audio_conv
                ch = int(model_channels * mult)
                if level == len(channel_mult)-2:   
                    layers.append(CrossAttentionBlock_cell(
                            ch,
                            feature_dim=feature_dim,
                            use_checkpoint=use_checkpoint,
                            local_window=self.cross_attention_windows[0],
                            window_shift=self.cross_attention_shift, 
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            id=3,
                            specific_type=self.specific_type,
                    ))
                
                bid += 1
                self._feature_size += ch
                self.output_blocks.append(TimestepEmbedSequential(*layers))

        
        self.audio_out = nn.Sequential(
            normalization_cell(input_ch),
            nn.SiLU(),
            CellConv(input_ch, audio_out_channels, feature_dim, kernel_size=3, conv_type='1d'),
        )
        self.video_out = nn.Sequential(
            normalization_cell(input_ch),
            nn.SiLU(),
            CellConv(input_ch, video_out_channels, feature_dim, kernel_size=3, conv_type='1d'),
        )
    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        # self.linear_in_audio.apply(convert_module_to_f16)
        # self.linear_in_video.apply(convert_module_to_f16)
        # self.linear_out_audio.apply(convert_module_to_f16)
        # self.linear_out_video.apply(convert_module_to_f16)
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_blocks.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)
        self.video_out.apply(convert_module_to_f16)
        self.audio_out.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.linear_in_audio.apply(convert_module_to_f32)
        self.linear_in_video.apply(convert_module_to_f32)
        self.linear_out_audio.apply(convert_module_to_f32)
        self.linear_out_video.apply(convert_module_to_f32)
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_blocks.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)
        self.video_out.apply(convert_module_to_f32)
        self.audio_out.apply(convert_module_to_f32)

    def load_state_dict_(self, state_dict, is_strict=False):
              
        for key, val in self.state_dict().items():
            
            if key in state_dict.keys():
                if val.shape == state_dict[key].shape:
                    continue
                else:
                    state_dict.pop(key)
                    logger.log("{} not matchable with state_dict with shape {}".format(key, val.shape))
            else:
                
                logger.log("{} not exists in state_dict".format(key))

        for key, val in state_dict.items():
            if key in self.state_dict().keys():
                if val.shape == state_dict[key].shape:
                    continue  
            else:
                logger.log("{} not used in state_dict".format(key))
        self.load_state_dict(state_dict, strict=is_strict)
        return 
    
   

    def forward(self, video, audio,  timesteps,  label=None, return_attvec=False):
        """
        Apply the model to an input batch.
        :param video: an [N x F x C x H x W] Tensor of inputs.
        :param audio: an [N x C x L] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param label: an [N] Tensor of labels, if class-conditional.
        :return: a video output of [N x F x C x H x W] Tensor, an audio output of [N x C x L] 
        """
        
       
        att_vectors = []
        video_hs = []
        audio_hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels)) # 
    
        if self.num_classes is not None:
            assert label.shape[0] == (video.shape[0])
            emb = emb + self.label_emb(label)

        video = video.type(self.dtype)
        audio = audio.type(self.dtype)
        emb = emb.type(self.dtype)

        for m_id, module in enumerate(self.input_blocks):# 
            if isinstance(module[-1], CrossAttentionBlock_cell):
                module[-1].specific_type = self.specific_type
                video, audio, rna_vector, atac_vector, att_vector1, att_vector2 = module(video, audio, emb)
                att_vectors.append(rna_vector)
                att_vectors.append(atac_vector)
                att_vectors.append(att_vector1)
                att_vectors.append(att_vector2)
            else:
                video, audio = module(video, audio, emb)#
            video_hs.append(video)
            audio_hs.append(audio)

        self.middle_blocks[0].specific_type = self.specific_type
        video, audio, rna_vector, atac_vector, att_vector1, att_vector2 = self.middle_blocks(video, audio, emb)#
        att_vectors.append(rna_vector)
        att_vectors.append(atac_vector)
        att_vectors.append(att_vector1)
        att_vectors.append(att_vector2)

        for m_id, module in enumerate(self.output_blocks):
            video = th.cat([video, video_hs.pop()], dim=-1)
            audio = th.cat([audio, audio_hs.pop()], dim=-1)
            if isinstance(module[-1], CrossAttentionBlock_cell):
                module[-1].specific_type = self.specific_type
                video, audio, rna_vector, atac_vector, att_vector1, att_vector2 = module(video, audio, emb)
                att_vectors.append(rna_vector)
                att_vectors.append(atac_vector)
                att_vectors.append(att_vector1)
                att_vectors.append(att_vector2)
            else:
                video, audio = module(video, audio, emb)#
       
        video = self.video_out(video)
        audio = self.audio_out(audio)
    
        if return_attvec:
            return video, audio, att_vectors
        
        return video, audio



if __name__=='__main__':
    import time
    device = th.device("cuda:7")
    

    model_channels = 192
    emb_channels = 128
    rna_dim= [16,3,64,64]
    atac_dim = [1, 25600]
    video_out_channels = 3
    audio_out_channels = 1
    num_heads = 2
    num_res_blocks = 1
    cross_attention_resolutions = [4,8,16]
    cross_attention_window = [1,1,1]
    cross_attention_shift = False
    video_attention_resolutions = [2,4,8,16]
    audio_attention_resolutions = [2,4,8,16]
    lr=0.0001
    channel_mult=(1,2,3,4)
    model = MultimodalUNet(
        rna_dim,
        atac_dim,
        model_channels,
        video_out_channels,
        audio_out_channels,
        num_res_blocks,
        cross_attention_resolutions = cross_attention_resolutions,
        num_heads = num_heads,
        cross_attention_windows = cross_attention_window,
        cross_attention_shift = cross_attention_shift,
        video_attention_resolutions = video_attention_resolutions,
        audio_attention_resolutions = audio_attention_resolutions,
        use_scale_shift_norm=True,
        use_checkpoint=True
        
    ).to(device)
    
    optim = th.optim.SGD(model.parameters(),lr=lr)
    model.train()
    while True:
        time_start=time.time()
        video = th.randn([1, 16, 3, 64, 64]).to(device)
        audio = th.randn([1, 1, 25600]).to(device)
        time_index = th.tensor([1]).to(device)

        video_out, audio_out = model(video, audio, time_index)
        video_target = th.randn_like(video_out)
        audio_target = th.randn_like(audio_out)
        loss =  F.mse_loss(video_target, video_out)+F.mse_loss(audio_target, audio_out)
        optim.zero_grad()
        loss.backward()
        optim.step()
        print(f"loss:{loss} time:{time.time()-time_start}")
  