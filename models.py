import yaml
import random
import inspect
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import repeat
from tools.torch_tools import wav_to_fbank

from audioldm.audio.stft import TacotronSTFT
from audioldm.variational_autoencoder import AutoencoderKL
from audioldm.utils import default_audioldm_config, get_metadata

from transformers import CLIPTokenizer, AutoTokenizer  #, T5Tokenizer
from transformers import CLIPTextModel, T5EncoderModel, AutoModel
from transformers import AlbertConfig, AlbertModel, AlbertTokenizer
from transformers import BertConfig, BertModel, TFBertTokenizer
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer
from transformers import T5Tokenizer, T5ForConditionalGeneration

from diffusers import StableDiffusionPipeline

import diffusers
from diffusers.utils import randn_tensor
from diffusers import DDPMScheduler, UNet2DConditionModel
from diffusers import AutoencoderKL as DiffuserAutoencoderKL

import copy

from collections import OrderedDict

import torch
from transformers import BlipForConditionalGeneration, BlipProcessor
from diffusers import DDIMScheduler, DDIMInverseScheduler, StableDiffusionPix2PixZeroPipeline
from diffusers import StableDiffusionPipeline, UNet2DConditionModel

import requests
from PIL import Image

import accelerate

# unet_model_config_path = "configs/stable_diffusion_2.1.json"
# accelerate.utils.set_seed(0)

# captioner_id = "Salesforce/blip-image-captioning-base"
# processor = BlipProcessor.from_pretrained(captioner_id)
# model = BlipForConditionalGeneration.from_pretrained(
#     captioner_id, low_cpu_mem_usage=True
# )

sd_model_ckpt = 'stabilityai/stable-diffusion-2-1'

# generate image latents

def get_image_latents(mdl, image, sample=True, rng_generator=None):
    encoding_dist = mdl.vae.encode(image).latent_dist
    if sample:
        encoding = encoding_dist.sample(generator=rng_generator)
    else:
        encoding = encoding_dist.mode()
    latents = encoding # + torch.randn_like(encoding) # * 0.18215
    return latents

def load_model(ckpt_path, pretrained=True):
    if ckpt_path is None and pretrained is True:
        raise NameError('Missing checkpoint path')
    elif ckpt_path is None:
        return None
    pretrained_weights = torch.jit.load(ckpt_path)
    new_dict = OrderedDict()
    for k,v in pretrained_weights.state_dict().items():
        if 'visual' in k:
            new_dict[k.replace('visual.','')] = v
    return new_dict

# from tango import Tango

def build_pretrained_models(name):

    # import pdb; pdb.set_trace()
    checkpoint = torch.load(get_metadata()[name]["path"], map_location="cpu")
    scale_factor = checkpoint["state_dict"]["scale_factor"].item()

    vae_state_dict = {k[18:]: v for k, v in checkpoint["state_dict"].items() if "first_stage_model." in k}

    config = default_audioldm_config(name)
    vae_config = config["model"]["params"]["first_stage_config"]["params"]
    vae_config["scale_factor"] = scale_factor

    vae = AutoencoderKL(**vae_config)
    vae.load_state_dict(vae_state_dict)

    fn_STFT = TacotronSTFT(
        config["preprocessing"]["stft"]["filter_length"],
        config["preprocessing"]["stft"]["hop_length"],
        config["preprocessing"]["stft"]["win_length"],
        config["preprocessing"]["mel"]["n_mel_channels"],
        config["preprocessing"]["audio"]["sampling_rate"],
        config["preprocessing"]["mel"]["mel_fmin"],
        config["preprocessing"]["mel"]["mel_fmax"],
    )

    vae.eval()
    fn_STFT.eval()
    return vae, fn_STFT


class AudioDiffusion(nn.Module):
    def __init__(
        self,
        text_encoder_name,
        scheduler_name,
        unet_model_name=None,
        unet_model_config_path=None,
        snr_gamma=None,
        freeze_text_encoder=True,
        uncondition=False,
        train_type="combined",
        args=None

    ):
        super().__init__()
        
        unet_model_config_path_sd = "configs/stable_diffusion_2.1.json"
        
        unet_config_sd = UNet2DConditionModel.load_config(unet_model_config_path_sd)

        self.sd_pipe = StableDiffusionPipeline.from_pretrained(sd_model_ckpt, use_auth_token='', low_cpu_mem_usage=False, device_map=None) # , scheduler=pipeline.scheduler)
        self.sd_pipe.unet = UNet2DConditionModel.from_config(unet_config_sd, subfolder="unet")
        self.sd_pipe.unet.load_state_dict(torch.load('sd_ckpt/sd_ckpt.pth', map_location='cpu')['unet_state_dict'], strict=False)
        # sd_pipe = sd_pipe.to("cuda")
        
        self.args = args
        
        if self.args is None:
            self.train_unet_img = False
        else:
            self.train_unet_img = self.args.train_unet_img

        assert unet_model_name is not None or unet_model_config_path is not None, "Either UNet pretrain model name or a config file path is required"
        
        self.text_encoder_name = text_encoder_name
        self.scheduler_name = scheduler_name
        self.unet_model_name = unet_model_name   
        
        self.unet_model_config_path = unet_model_config_path
        self.snr_gamma = snr_gamma
        self.freeze_text_encoder = freeze_text_encoder
        self.uncondition = uncondition

        self.noise_scheduler = DDPMScheduler.from_pretrained(self.scheduler_name, subfolder="scheduler")
        self.inference_scheduler = DDPMScheduler.from_pretrained(self.scheduler_name, subfolder="scheduler")
        self.train_type = train_type
        
        if unet_model_config_path:
            unet_config = UNet2DConditionModel.load_config(unet_model_config_path)
            
            self.unet = UNet2DConditionModel.from_config(unet_config, subfolder="unet")
            for params in self.unet.parameters():
                params.requires_grad = True # False

            self.set_from = "random"
            print("UNet initialized randomly.")
            
        else:
            self.unet = UNet2DConditionModel.from_pretrained(unet_model_name, subfolder="unet")
            self.set_from = "pre-trained"
            self.group_in = nn.Sequential(nn.Linear(8, 512), nn.Linear(512, 4))
            self.group_out = nn.Sequential(nn.Linear(4, 512), nn.Linear(512, 8))
            print("UNet initialized from stable diffusion checkpoint.")
        
        if "stable-diffusion" in self.text_encoder_name:
            self.tokenizer = CLIPTokenizer.from_pretrained(self.text_encoder_name, subfolder="tokenizer")
            self.text_encoder = CLIPTextModel.from_pretrained(self.text_encoder_name, subfolder="text_encoder", cache_dir="/fs/nexus-scratch/sanjoyc/.cache")
            print('loading CLIP text encoder')
        
        elif "t5" in self.text_encoder_name:
            self.tokenizer = AutoTokenizer.from_pretrained(self.text_encoder_name)   
            self.text_encoder = T5EncoderModel.from_pretrained(self.text_encoder_name, cache_dir="/fs/nexus-scratch/sanjoyc/.cache")
            
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.text_encoder_name)   
            self.text_encoder = T5EncoderModel.from_pretrained(self.text_encoder_name, cache_dir="/fs/nexus-scratch/sanjoyc/.cache")
            
    def compute_snr(self, timesteps):
        """
        Computes SNR as per https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
        """
        alphas_cumprod = self.noise_scheduler.alphas_cumprod
        sqrt_alphas_cumprod = alphas_cumprod**0.5
        sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

        # Expand the tensors.
        # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
        sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
        while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
        alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
        while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
        sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

        # Compute SNR.
        snr = (alpha / sigma) ** 2
        return snr

    def encode_text(self, prompt):
        device = self.text_encoder.device
        batch = self.tokenizer(
            prompt, max_length=self.tokenizer.model_max_length, padding=True, truncation=True, return_tensors="pt"
        )
        input_ids, attention_mask = batch.input_ids.to(device), batch.attention_mask.to(device)

        if self.freeze_text_encoder:
            with torch.no_grad():
                encoder_hidden_states = self.text_encoder(
                    input_ids=input_ids, attention_mask=attention_mask
                )[0]
        else:
            encoder_hidden_states = self.text_encoder(
                input_ids=input_ids, attention_mask=attention_mask
            )[0]

        boolean_encoder_mask = (attention_mask == 1).to(device)
        return encoder_hidden_states, boolean_encoder_mask

    def forward(self, latents, prompt, visual_imgs, validation_mode=False):
        device = self.text_encoder.device
        num_train_timesteps = self.noise_scheduler.num_train_timesteps
        self.noise_scheduler.set_timesteps(num_train_timesteps, device=device)
        
        visual_imgs = visual_imgs.squeeze(1)
        
        encoder_hidden_states, boolean_encoder_mask = self.encode_text(prompt)
        
        if self.uncondition:
            mask_indices = [k for k in range(len(prompt)) if random.random() < 0.1]
            if len(mask_indices) > 0:
                encoder_hidden_states[mask_indices] = 0

        bsz = latents.shape[0]

        if validation_mode:
            timesteps = (self.noise_scheduler.num_train_timesteps//2) * torch.ones((bsz,), dtype=torch.int64, device=device)
        else:
            # Sample a random timestep for each instance
            timesteps = torch.randint(0, self.noise_scheduler.num_train_timesteps, (bsz,), device=device)
        timesteps = timesteps.long()

        noise = torch.randn_like(latents)
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        
        # latents_img = torch.randn(bsz, 4, 64, 64).to(latents.device)
        latents_img = get_image_latents(self.sd_pipe.to(latents.device), visual_imgs.to(latents.device), rng_generator=torch.Generator(device=latents.device).manual_seed(0))
        if latents.shape[0] - latents_img.shape[0] == 1:
            latents_img = torch.cat((latents_img,0.5*latents_img[0:1,:] + 0.5*latents_img[1:2,:]),dim=0)
        else:
            temp_factor = latents.shape[0] // latents_img.shape[0]
            latents_img = latents_img.repeat_interleave(temp_factor, 0)
        noisy_latents_img = self.noise_scheduler.add_noise(latents_img, torch.randn_like(latents_img), timesteps)

        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")

        if self.set_from == "random":
            self.sd_pipe = self.sd_pipe.to(self.unet.device)
            noise_pred, cross_attn = self.sd_pipe.unet(
                noisy_latents_img,
                timesteps,
                encoder_hidden_states,
                external_hidden_states = None,
                encoder_attention_mask=None,
            ) # .sample
            model_pred, _ = self.unet(
                noisy_latents, timesteps, encoder_hidden_states, external_hidden_states=cross_attn,
                encoder_attention_mask=boolean_encoder_mask
            ) # .sample
            
            # print("@@@@@@ cross_attn.shape", cross_attn['1'][0][0].shape)
            # print("@@@@@@ ca_.shape", ca_['1'][0][0].shape)
            
        
        elif self.set_from == "pre-trained":
            compressed_latents = self.group_in(noisy_latents.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous()
            self.sd_pipe = self.sd_pipe.to(self.unet.device)
            noise_pred, cross_attn = self.sd_pipe.unet(
                noisy_latents_img,
                timesteps,
                encoder_hidden_states,
                external_hidden_states = None,
                encoder_attention_mask=None,
            )
            model_pred, _ = self.unet(
                compressed_latents, timesteps, encoder_hidden_states, external_hidden_states=cross_attn,
                encoder_attention_mask=boolean_encoder_mask
            ) # .sample
            model_pred = self.group_out(model_pred.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous()
                        
        target = target.mean(0).unsqueeze(0)    
        model_pred = model_pred.mean(0).unsqueeze(0)
        
        if self.snr_gamma is None:
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        else:
            # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
            # Adaptef from huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image.py
            snr = self.compute_snr(timesteps)
            mse_loss_weights = (
                torch.stack([snr, self.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
            )
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
            loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
            loss = loss.mean()

        total_loss = loss
            
        if validation_mode:
            return total_loss
        else:
            return total_loss 

    
    @torch.no_grad()
    def inference(self, prompt, img_tensors, inference_scheduler, num_steps=20, guidance_scale=3, num_samples_per_prompt=1, 
                  disable_progress=True, inference_choice='text'):
        device = self.text_encoder.device
        classifier_free_guidance = guidance_scale > 1.0
        batch_size = len(prompt) * num_samples_per_prompt
        
        self.sd_pipe = self.sd_pipe.to(device)
                       
        if classifier_free_guidance:
            prompt_embeds, boolean_prompt_mask = self.encode_text_classifier_free(prompt, num_samples_per_prompt)
        else:
            prompt_embeds, boolean_prompt_mask = self.encode_text(prompt)
            prompt_embeds = prompt_embeds.repeat_interleave(num_samples_per_prompt, 0)
            boolean_prompt_mask = boolean_prompt_mask.repeat_interleave(num_samples_per_prompt, 0)
        
        inference_scheduler.set_timesteps(num_steps, device=device)
        timesteps = inference_scheduler.timesteps
        
        self.sd_pipe.scheduler.set_timesteps(num_steps, device=device)

        num_channels_latents = self.unet.in_channels
        latents = self.prepare_latents(batch_size, inference_scheduler, num_channels_latents, prompt_embeds.dtype, device)
        
        height = self.sd_pipe.unet.config.sample_size * self.sd_pipe.vae_scale_factor
        width = self.sd_pipe.unet.config.sample_size * self.sd_pipe.vae_scale_factor
        latents_img = self.sd_pipe.prepare_latents(
            len(prompt) * 1,
            self.sd_pipe.unet.in_channels,
            height,
            width,
            prompt_embeds.dtype,
            device,
            None,
            torch.randn(len(prompt),4,64,64),
        )

        num_warmup_steps = len(timesteps) - num_steps * inference_scheduler.order
        progress_bar = tqdm(range(num_steps), disable=disable_progress)

        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if classifier_free_guidance else latents
            latent_model_input = inference_scheduler.scale_model_input(latent_model_input, t)
            
            latent_model_input_img = torch.cat([latents_img] * 2) if classifier_free_guidance else latents_img
            latent_model_input_img = self.sd_pipe.scheduler.scale_model_input(latent_model_input_img, t)
            
            noise_pred_img, cross_attn = self.sd_pipe.unet(
                latent_model_input_img,
                t,
                encoder_hidden_states=prompt_embeds,
                external_hidden_states = None,
                encoder_attention_mask=None,
            ) # .sample
            
            # if prompt_embeds.shape[-1] == 768:
            #     prompt_embeds = torch.nn.AdaptiveAvgPool1d(1024)(prompt_embeds) 
                
            if(inference_choice == "text"):
                # print('inference_choice is: ', inference_choice)
                noise_pred, _ = self.unet(
                    latent_model_input, t, encoder_hidden_states=prompt_embeds,
                    external_hidden_states = None,
                    encoder_attention_mask=boolean_prompt_mask
                ) # .sample
            
                
            elif(inference_choice == "image"):
                noise_pred, _ = self.unet(
                    latent_model_input, t, encoder_hidden_states=prompt_embeds,
                    external_hidden_states = cross_attn,
                    encoder_attention_mask=boolean_prompt_mask
                ) # .sample
            

            # perform guidance
            if classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                
                noise_pred_img_uncond, noise_pred_img_text = noise_pred_img.chunk(2)
                noise_pred_img = noise_pred_img_uncond + guidance_scale * (noise_pred_img_text - noise_pred_img_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = inference_scheduler.step(noise_pred, t, latents).prev_sample
            
            latents_img = self.sd_pipe.scheduler.step(noise_pred_img, t, latents_img).prev_sample

            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % inference_scheduler.order == 0):
                progress_bar.update(1)

        if self.set_from == "pre-trained":
            latents = self.group_out(latents.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous()
        return latents

    def prepare_latents(self, batch_size, inference_scheduler, num_channels_latents, dtype, device):
        shape = (batch_size, num_channels_latents, 256, 16) # maybe change this 256 to 512 to make it 20 secs
        latents = randn_tensor(shape, generator=None, device=device, dtype=dtype)
        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * inference_scheduler.init_noise_sigma
        return latents

    def encode_text_classifier_free(self, prompt, num_samples_per_prompt):
        device = self.text_encoder.device
        batch = self.tokenizer(
            prompt, max_length=self.tokenizer.model_max_length, padding=True, truncation=True, return_tensors="pt"
        )
        input_ids, attention_mask = batch.input_ids.to(device), batch.attention_mask.to(device)

        with torch.no_grad():
            prompt_embeds = self.text_encoder(
                input_ids=input_ids, attention_mask=attention_mask
            )[0]
                
        prompt_embeds = prompt_embeds.repeat_interleave(num_samples_per_prompt, 0)
        attention_mask = attention_mask.repeat_interleave(num_samples_per_prompt, 0)

        # get unconditional embeddings for classifier free guidance
        uncond_tokens = [""] * len(prompt)

        max_length = prompt_embeds.shape[1]
        uncond_batch = self.tokenizer(
            uncond_tokens, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt",
        )
        uncond_input_ids = uncond_batch.input_ids.to(device)
        uncond_attention_mask = uncond_batch.attention_mask.to(device)

        with torch.no_grad():
            negative_prompt_embeds = self.text_encoder(
                input_ids=uncond_input_ids, attention_mask=uncond_attention_mask
            )[0]
                
        negative_prompt_embeds = negative_prompt_embeds.repeat_interleave(num_samples_per_prompt, 0)
        uncond_attention_mask = uncond_attention_mask.repeat_interleave(num_samples_per_prompt, 0)

        # For classifier free guidance, we need to do two forward passes.
        # We concatenate the unconditional and text embeddings into a single batch to avoid doing two forward passes
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
        prompt_mask = torch.cat([uncond_attention_mask, attention_mask])
        boolean_prompt_mask = (prompt_mask == 1).to(device)

        return prompt_embeds, boolean_prompt_mask
    

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)
    
    
    
class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)
    
    
    
class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

    
    
    

class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, :, :])

        if self.proj is not None:
            x = x @ self.proj
            
        x = torch.nn.AdaptiveAvgPool1d(1024)(x)

        return x[:,1:,:], x[:,0:1,:]


        
        
