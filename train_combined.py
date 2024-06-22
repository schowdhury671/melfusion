import time
import argparse
import json
import logging
import math
import os
# from tqdm import tqdm
from pathlib import Path
import subprocess as sp

import datasets
import numpy as np
import pandas as pd
import wandb
import torch
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

import soundfile as sf
import diffusers
import transformers
import tools.torch_tools as torch_tools
from huggingface_hub import snapshot_download
from models import build_pretrained_models, AudioDiffusion, VisionTransformer
from transformers import SchedulerType, get_scheduler
from PIL import Image
import torchvision.transforms as transforms
import pickle

logger = get_logger(__name__)


def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_values


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a diffusion model for text to audio generation task.")
    parser.add_argument(
        "--train_file", type=str, default="data/musiccaps_with_valid_image_audio_full.json",
        help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default="data/musiccaps_with_images_valid_small.json",
        help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--test_file", type=str, default="data/musiccaps_with_images_valid_small.json",
        help="A csv or a json file containing the test data for generation."
    )     
    parser.add_argument(
        "--num_examples", type=int, default=30000,
        help="How many examples to use for training and validation.",
    )
    parser.add_argument(
        "--text_encoder_name", type=str, default="google/flan-t5-large",
        help="Text encoder identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--scheduler_name", type=str, default="stabilityai/stable-diffusion-2-1",
        help="Scheduler identifier.",
    )
    parser.add_argument(
        "--unet_model_name", type=str, default=None,
        help="UNet model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--unet_model_config", type=str, default=None,
        help="UNet model config json path.",
    )
    parser.add_argument(
        "--hf_model", type=str, default=None,
        help="Tango model identifier from huggingface: declare-lab/tango",
    )
    parser.add_argument(
        "--snr_gamma", type=float, default=None,
        help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. "
        "More details here: https://arxiv.org/abs/2303.09556.",
    )
    parser.add_argument(
        "--train_unet_img", action="store_true", default=False,
        help="Train unet image model or not.",
    )
    parser.add_argument(
        "--train_only_alpha", action="store_true", default=False,
        help="Train only alpha or not.",
    )
    parser.add_argument(
        "--freeze_text_encoder", action="store_true", default=False,
        help="Freeze the text encoder model.",
    )
    parser.add_argument(
        "--text_column", type=str, default="captions",
        help="The name of the column in the datasets containing the input texts.",
    )
    parser.add_argument(
        "--audio_column", type=str, default="location",
        help="The name of the column in the datasets containing the audio paths.",
    )
    parser.add_argument(
        "--img_column", type=str, default="img_path",
        help="The name of the column in the datasets containing the image paths.",
    )
    parser.add_argument(
        "--augment", action="store_true", default=False,
        help="Augment training data.",
    )
    parser.add_argument(
        "--uncondition", action="store_true", default=False,
        help="10% uncondition for training.",
    )
    parser.add_argument(
        "--prefix", type=str, default=None,
        help="Add prefix in text prompts.",
    )
    parser.add_argument(
        "--per_device_train_batch_size", type=int, default=2,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size", type=int, default=2,
        help="Batch size (per device) for the validation dataloader.",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=1e-8,
        help="Weight decay to use."
    )
    parser.add_argument(
        "--num_train_epochs", type=int, default=500,
        help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_train_steps", type=int, default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps", type=int, default=4,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type", type=SchedulerType, default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0,
        help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--adam_beta1", type=float, default=0.9,
        help="The beta1 parameter for the Adam optimizer."
    )
    parser.add_argument(
        "--adam_beta2", type=float, default=0.999,
        help="The beta2 parameter for the Adam optimizer."
    )
    parser.add_argument(
        "--adam_weight_decay", type=float, default=1e-2,
        help="Weight decay to use."
    )
    parser.add_argument(
        "--adam_epsilon", type=float, default=1e-08,
        help="Epsilon value for the Adam optimizer"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Where to store the final model."
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="A seed for reproducible training."
    )
    parser.add_argument(
        "--checkpointing_steps", type=str, default="best",
        help="Whether the various states should be saved at the end of every 'epoch' or 'best' whenever validation loss decreases.",
    )
    parser.add_argument(
        "--save_every", type=int, default=10,
        help="Save model after every how many epochs when checkpointing_steps is set to best."
    )
    parser.add_argument(
        "--resume_from_checkpoint", type=str, default=None,
        help="If the training should continue from a local checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking", action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--train_type", type=str, default = "combined",
        help="Training type.",
    )
    parser.add_argument(
        "--report_to", type=str, default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    args = parser.parse_args()

    # Sanity checks
    if args.train_file is None and args.validation_file is None:
        raise ValueError("Need a training/validation file.")
    else:
        if args.train_file is not None:
        
            #print("@!@!@!@!@!@!@trying to read from train file ",args.train_file)
            extension = args.train_file.split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
        if args.validation_file is not None:
            #print("@!@!@!@!@!@!@trying to read from validation file ",args.validation_file)
            extension = args.validation_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."

    return args

class Text2AudioDataset(Dataset):
    def __init__(self, dataset, prefix, text_column, audio_column, img_column, num_examples=-1):

        # try:
        inputs = list(dataset[text_column])
        self.inputs = [prefix + inp for inp in inputs]
        self.audios = list(dataset[audio_column])
        self.imgs = list(dataset[img_column])
        
        # temp_inputs, temp_audios, temp_imgs = [], [], []
        # for a, b, c in zip(self.inputs, self.audios, self.imgs):
        #     if os.path.exists(c) is False:
        #         temp_inputs.append(a)
        #         temp_audios.append(b)
        #         temp_imgs.append(c)
        # self.inputs = temp_inputs
        # self.audios = temp_audios
        # self.imgs = temp_imgs
        # del temp_inputs
        # del temp_audios
        # del temp_imgs
        # print(len(self.inputs))
        
        self.indices = list(range(len(self.inputs)))

        self.mapper = {}
        for index, audio, text, img in zip(self.indices, self.audios, inputs, self.imgs):
            self.mapper[index] = [audio, text, img]
        # except:
        #     pass

        if num_examples != -1:
            self.inputs, self.audios, self.imgs = self.inputs[:num_examples], self.audios[:num_examples], self.imgs[:num_examples]
            self.indices = self.indices[:num_examples]

    def __len__(self):
        return len(self.inputs)

    def get_num_instances(self):
        return len(self.inputs)

    def __getitem__(self, index):
        
        # try:
        s1, s2, s3, s4 = self.inputs[index], self.audios[index], self.indices[index], self.imgs[index]

        image = Image.open(s4)
        image = image.resize((512,512))
        transform = transforms.Compose([transforms.PILToTensor()])
        img_tensor = transform(image)
        img_tensor = img_tensor.float()
        img_tensor = 2.0 * img_tensor - 1.0

        # print("s2 is ",s2)
        return s1, s2, s3, img_tensor.unsqueeze(0)
        # except:
        #     pass

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [dat[i].tolist() for i in dat]
    
    
def main():
    args = parse_args()
    accelerator_log_kwargs = {}

    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["logging_dir"] = args.output_dir
    ddp_kwargs = DistributedDataParallelKwargs({'static_graph':True}) # (find_unused_parameters=True)        
        
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, kwargs_handlers=[ddp_kwargs], **accelerator_log_kwargs)
    
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    datasets.utils.logging.set_verbosity_error()
    diffusers.utils.logging.set_verbosity_error()
    transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle output directory creation and wandb tracking
    if accelerator.is_main_process:
        if args.output_dir is None or args.output_dir == "":
            args.output_dir = "saved/" + str(int(time.time()))
            
            if not os.path.exists("saved"):
                os.makedirs("saved")
                
            os.makedirs(args.output_dir, exist_ok=True)
            
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        os.makedirs("{}/{}".format(args.output_dir, "outputs"), exist_ok=True)
        with open("{}/summary.jsonl".format(args.output_dir), "a") as f:
            f.write(json.dumps(dict(vars(args))) + "\n\n")

        accelerator.project_configuration.automatic_checkpoint_naming = False

        # wandb.init(project="Text to Audio Diffusion")

    accelerator.wait_for_everyone()

    # Get the datasets
    data_files = {}
    
    
    # print("@!@!@!@!train file is ",args.train_file)
    # print("@!@!@!@!validation file is ",args.validation_file)
    # print("@!@!@!@!test file is ",args.test_file)
    
    
    if args.train_file is not None:
        data_files["train"] = args.train_file
    if args.validation_file is not None:
        data_files["validation"] = args.validation_file

    if args.test_file is not None:
        data_files["test"] = args.test_file
    else:
        if args.validation_file is not None:
            data_files["test"] = args.validation_file

    extension = args.train_file.split(".")[-1]
    raw_datasets = load_dataset(extension, data_files=data_files)
    text_column, audio_column, img_column = args.text_column, args.audio_column, args.img_column

    # Initialize models
    pretrained_model_name = "audioldm-s-full"
    vae, stft = build_pretrained_models(pretrained_model_name)
    vae.eval()
    stft.eval()

    model = AudioDiffusion(
        args.text_encoder_name, args.scheduler_name, args.unet_model_name, args.unet_model_config, args.snr_gamma, args.freeze_text_encoder, args.uncondition, args.train_type, args
    )
    # visual_encoder = VisionTransformer(input_resolution = 224, patch_size = 16, width = 32, layers = 8, heads = 8, output_dim = 1024)
    
    if args.hf_model:
        hf_model_path = snapshot_download(repo_id=args.hf_model)
        model.load_state_dict(torch.load("{}/pytorch_model_main.bin".format(hf_model_path), map_location="cpu"),strict=False)
        accelerator.print("Successfully loaded checkpoint from:", args.hf_model)
        
        ######### if using own pre-trained model instead of huggingface model from tango ################
        
        # hf_model_path = args.hf_model
        # model.load_state_dict(torch.load(hf_model_path, map_location="cpu"),strict=False)
        # accelerator.print("Successfully loaded checkpoint from:", args.hf_model)
        
    # for params_text, params_img in zip(model.unet.parameters(), model.unet_img.parameters()):
    #     params_img.data.copy_(params_text.data)  # initialize
    #     params_img.requires_grad = False
    
    if args.prefix:
        prefix = args.prefix
    else:
        prefix = ""

    with accelerator.main_process_first():
        
        # print("dataset param is ", raw_datasets["train"])
        # print("prefix is ", prefix)
        # print("text col is ", text_column)
        # print("audio col is ", audio_column)
        # print("image column is ", img_column)
        # print("num_examples are ", args.num_examples)
        
        
        #print("train dataset is ", raw_datasets["train"][:20])
        
        
        train_dataset = Text2AudioDataset(raw_datasets["train"], prefix, text_column, audio_column, img_column, args.num_examples)
        eval_dataset = Text2AudioDataset(raw_datasets["validation"], prefix, text_column, audio_column, img_column, args.num_examples)
        test_dataset = Text2AudioDataset(raw_datasets["test"], prefix, text_column, audio_column, img_column, args.num_examples)

        accelerator.print("Num instances in train: {}, validation: {}, test: {}".format(train_dataset.get_num_instances(), eval_dataset.get_num_instances(), test_dataset.get_num_instances()))

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.per_device_train_batch_size, collate_fn=train_dataset.collate_fn)
    #print("@!@!@!@train_dataset first 20 entries are is ", train_dataset[:20])
    
    eval_dataloader = DataLoader(eval_dataset, shuffle=False, batch_size=args.per_device_eval_batch_size, collate_fn=eval_dataset.collate_fn)
    #print("@!@!@!@eval_dataset first 20 entries are is ", eval_dataloader[:20])
    
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=args.per_device_eval_batch_size, collate_fn=test_dataset.collate_fn)
    #print("@!@!@!@test_dataset first 20 entries are is ", test_dataloader[:20])
    
    # Optimizer
    if args.freeze_text_encoder:
        for param in model.text_encoder.parameters():
            param.requires_grad = False
            model.text_encoder.eval()
            # print("****************freeze text encoder")
        if args.unet_model_config:
            for param in model.unet.parameters():
                param.requires_grad = True # train Audio (Tango) U-Net
            
            for param in model.sd_pipe.unet.parameters():
                param.requires_grad = False # freeze Stable Diffusion U-Net
                
            if args.train_only_alpha:
                p_list = []
                for name, param in model.unet.named_parameters():
                    if 'alpha' in name:
                        if 'up' in name:
                            if 'attn2' in name:
                                p_list.append(param)
                            else:
                                param.requires_grad = False
                        else:
                            param.requires_grad = False
                optimizer_parameters = p_list # train only alpha factor
                accelerator.print("Training alphas only.")
            else:
                optimizer_parameters = list(model.unet.parameters()) # train full Audio (Tango) U-Net
                accelerator.print("Training full Audio (Tango) U-Net.")
            accelerator.print("Total #params going in optim: " + str(len(optimizer_parameters)))
        else:
            optimizer_parameters = list(model.unet.parameters()) + list(model.group_in.parameters()) + list(model.group_out.parameters())
            accelerator.print("Optimizing UNet and channel transformer parameters.")
    else:
        optimizer_parameters = model.parameters()
        accelerator.print("Optimizing Text Encoder and UNet parameters.")

    num_trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    accelerator.print("Num trainable parameters: {}".format(num_trainable_parameters))

    # print("***********optimizer parameters ", optimizer_parameters)
    
    optimizer = torch.optim.AdamW(
        optimizer_parameters, lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # Prepare everything with our `accelerator`.
    vae, stft, model, optimizer, lr_scheduler = accelerator.prepare(
        vae, stft, model, optimizer, lr_scheduler
    )
    
    # model = model.cuda() # accelerator.prepare_model(model.cuda(), device_placement=True)
    # device = torch.device("cuda:0")
    # model = model.to(device)

    train_dataloader, eval_dataloader, test_dataloader = accelerator.prepare(
        train_dataloader, eval_dataloader, test_dataloader
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers("text_to_audio_diffusion", experiment_config)

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)

    completed_steps = 0
    starting_epoch = 0
    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            accelerator.load_state(args.resume_from_checkpoint)
            # path = os.path.basename(args.resume_from_checkpoint)
            accelerator.print(f"Resumed from local checkpoint: {args.resume_from_checkpoint}")
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            # path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
            
    # Duration of the audio clips in seconds
    duration, best_loss = 10, np.inf

    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        total_loss, total_val_loss, total_val_loss_img = 0, 0, 0
        
        for step, batch in enumerate(train_dataloader):

            with accelerator.accumulate(model):
                device = model.device
                
                # try:
                text, audios, _, visual_imgs = batch
                for _i in range(len(visual_imgs)):
                    try:
                        assert visual_imgs[_i].shape[1] == 3

                    except:
                        #print("printing details: ",len(visual_imgs), visual_imgs[0].shape, visual_imgs[1].shape)
                        visual_imgs[_i] = visual_imgs[_i].repeat(1,3,1,1) 
                visual_imgs = torch.stack(visual_imgs, dim = 0)
                # except:
                #     pass
                    
                # print("visual images shape: ", len(visual_imgs), visual_imgs[0].shape)
                # visual_features = visual_encoder(visual_imgs)
                # visual_features = visual_features[0]
                # print(visual_features.shape)
                # print(abcd)
                
                target_length = int(duration * 102.4)

                # try:
                with torch.no_grad():
                    unwrapped_vae = accelerator.unwrap_model(vae)

                    mel, _, waveform = torch_tools.wav_to_fbank(audios, target_length, stft)
                    mel = mel.unsqueeze(1).to(device)

                    if args.augment:
                        mixed_mel, _, _, mixed_captions = torch_tools.augment_wav_to_fbank(audios, text, len(audios), target_length, stft)
                        mixed_mel = mixed_mel.unsqueeze(1).to(device)
                        mel = torch.cat([mel, mixed_mel], 0)
                        text += mixed_captions

                    true_latent = unwrapped_vae.get_first_stage_encoding(unwrapped_vae.encode_first_stage(mel))


                loss = model(true_latent, text, visual_imgs, validation_mode=False)
                total_loss += loss.detach().float()

                accelerator.backward(loss)
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1

            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0:
                    output_dir = f"step_{completed_steps }"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir)

            if completed_steps >= args.max_train_steps:
                break

        model.eval()
        model.uncondition = False

        eval_progress_bar = tqdm(range(len(eval_dataloader)), disable=not accelerator.is_local_main_process)
        for step, batch in enumerate(eval_dataloader):
            with accelerator.accumulate(model) and torch.no_grad():
                
                # try: 
                device = model.device
                text, audios, _ , visual_imgs = batch

                # print(len(text))

                for _i in range(len(visual_imgs)):
                    try:
                        assert visual_imgs[_i].shape[1] == 3

                    except:
                        #print("printing details: ",len(visual_imgs), visual_imgs[0].shape, visual_imgs[1].shape)
                        visual_imgs[_i] = visual_imgs[_i].repeat(1,3,1,1) 
                visual_imgs = torch.stack(visual_imgs, dim = 0)
                # except:
                #     pass
                
                target_length = int(duration * 102.4)

                unwrapped_vae = accelerator.unwrap_model(vae)
                
                # try:
                mel, _, waveform = torch_tools.wav_to_fbank(audios, target_length, stft)
                mel = mel.unsqueeze(1).to(device)
                true_latent = unwrapped_vae.get_first_stage_encoding(unwrapped_vae.encode_first_stage(mel))

                val_loss = model(true_latent, text, visual_imgs, validation_mode=True)
                val_loss_img = val_loss
                total_val_loss += val_loss.detach().float()
                total_val_loss_img += val_loss_img.detach().float()
                eval_progress_bar.update(1)
                # except:
                #     pass
        model.uncondition = args.uncondition

        if accelerator.is_main_process:    
            result = {}
            result["epoch"] = epoch+1,
            result["step"] = completed_steps
            
            # try:
            result["train_loss"] = round(total_loss.item()/len(train_dataloader), 4)
            result["val_loss"] = round(total_val_loss.item()/len(eval_dataloader), 4)
            result["val_loss_img"] = round(total_val_loss_img.item()/len(eval_dataloader), 4)
            result["len_dataloader"] = len(train_dataloader)
            # except:
            #     pass
            # wandb.log(result)

            # try:
            result_string = "Epoch: {}, Loss Train: {}, Val_text: {}, Val_img: {}, Train_loader Length: {}\n".format(epoch, result["train_loss"], result["val_loss"], result["val_loss_img"], len(train_dataloader))

            accelerator.print(result_string)

            with open("{}/summary.jsonl".format(args.output_dir), "a") as f:
                f.write(json.dumps(result) + "\n\n")

            logger.info(result)
            
            ### need to change based on what we want to validate for. for run 29 and 30 validation is done for image.. might required to be changed for text later on

            if result["val_loss_img"] < best_loss:
                best_loss = result["val_loss_img"]
                save_checkpoint = True
            else:
                save_checkpoint = False
            # except:
            #     pass
            
        # try:    
        if args.with_tracking:
            accelerator.log(result, step=completed_steps)

        accelerator.wait_for_everyone()
        if accelerator.is_main_process and args.checkpointing_steps == "best":
            if save_checkpoint:
                accelerator.save_state("{}/{}".format(args.output_dir, "best"))

            if (epoch + 1) % args.save_every == 0:
                accelerator.save_state("{}/{}".format(args.output_dir, "epoch_" + str(epoch+1)))

        if accelerator.is_main_process and args.checkpointing_steps == "epoch":
            accelerator.save_state("{}/{}".format(args.output_dir, "epoch_" + str(epoch+1)))
        # except:
        #     pass
            
if __name__ == "__main__":
    main()
