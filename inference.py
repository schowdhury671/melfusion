import os
import copy
import json
import time
import torch
import argparse
import soundfile as sf
import wandb
from tqdm import tqdm
from diffusers import DDPMScheduler
from audioldm_eval import EvaluationHelper
from models import build_pretrained_models, AudioDiffusion
from transformers import AutoProcessor, ClapModel
import torchaudio
from tango import Tango
from PIL import Image
import torchvision.transforms as transforms


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    
def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def parse_args():
    parser = argparse.ArgumentParser(description="Inference for text to audio generation task.")
    parser.add_argument(
        "--original_args", type=str, default=None,
        help="Path for summary jsonl file saved during training."
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Path for saved model bin file."
    )
    parser.add_argument(
        "--test_file", type=str, default="data/musiccaps-w-img-6-samples.json",
        help="json file containing the test prompts for generation."
    )
    parser.add_argument(
        "--text_key", type=str, default="captions",
        help="Key containing the text in the json file."
    )
    parser.add_argument(
        "--img_key", type=str, default="img_path",
        help="The name of the column in the datasets containing the image paths.",
    )
    parser.add_argument(
        "--test_references", type=str, default="data/audiocaps_test_references/subset",
        help="Folder containing the test reference wav files."
    )
    parser.add_argument(
        "--num_steps", type=int, default=200,
        help="How many denoising steps for generation.",
    )
    parser.add_argument(
        "--guidance", type=float, default=3,
        help="Guidance scale for classifier free guidance."
    )
    parser.add_argument(
        "--batch_size", type=int, default=1,
        help="Batch size for generation.",
    )
    parser.add_argument(
        "--num_samples", type=int, default=1,
        help="How many samples per prompt.",
    )
    parser.add_argument(
        "--num_test_instances", type=int, default=-1,
        help="How many test instances to evaluate.",
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
        "--output_pth", type=str, default=None,
        help="The output path where the results will be stored.",
    )
    parser.add_argument(
        "--inference_choice", type=str, default="text",
        help="pass 'text'for Text inference or 'image' for Image inference ",
    )
    
    
    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    
    train_args = args   # dotdict(json.loads(open(args.original_args).readlines()[0]))
    
    print("train_args.unet_model_config is ", train_args.unet_model_config)
    
    #train_args.hf_model = False
    
    output_folder_path = train_args.output_pth
    
    inference_choice = train_args.inference_choice
    
    if "hf_model" not in train_args:
        train_args["hf_model"] = None
    
    #print("hf_model param2 is ", train_args.hf_model)
    
    
    # Load Models #
    # train_args.hf_model = True
    
    if train_args.hf_model:
        print("loading hf model")
        # print(abcd)
        tango = Tango(train_args.hf_model, "cpu")    
        vae, stft, model = tango.vae.cuda(), tango.stft.cuda(), tango.model.cuda()
    else:
        print("loading audioldm model")
        # print(abcd)
        name = "audioldm-s-full"
        vae, stft = build_pretrained_models(name)
        vae, stft = vae.cuda(), stft.cuda()
        model = AudioDiffusion(
            train_args.text_encoder_name, train_args.scheduler_name, train_args.unet_model_name, train_args.unet_model_config, train_args.snr_gamma
        ).cuda()
    
    model.eval()  # this was inside else loop
    
    # Load Trained Weight #
    device = vae.device()
    
    model.load_state_dict(torch.load(args.model), strict=False) # model.load_state_dict(torch.load("saved/pytorch_model_main.bin")['state_dict'],strict=False)
    # import pdb; pdb.set_trace()
    
    # model.load_state_dict(torch.load("saved/pytorch_model_main.bin")['state_dict'],strict=False)
    
    scheduler = DDPMScheduler.from_pretrained(train_args.scheduler_name, subfolder="scheduler")
    evaluator = EvaluationHelper(16000, "cuda:0")
    
    if args.num_samples > 1:
        clap = ClapModel.from_pretrained("laion/clap-htsat-unfused").to(device)
        clap.eval()
        clap_processor = AutoProcessor.from_pretrained("laion/clap-htsat-unfused")
    
    wandb.init(project="Text + Image to Audio Diffusion Evaluation")

    def audio_text_matching(waveforms, text, sample_freq=16000, max_len_in_seconds=10):
        new_freq = 48000
        resampled = []
        
        for wav in waveforms:
            x = torchaudio.functional.resample(torch.tensor(wav, dtype=torch.float).reshape(1, -1), orig_freq=sample_freq, new_freq=new_freq)[0].numpy()
            resampled.append(x[:new_freq*max_len_in_seconds])

        inputs = clap_processor(text=text, audios=resampled, return_tensors="pt", padding=True, sampling_rate=48000)  
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = clap(**inputs)

        logits_per_audio = outputs.logits_per_audio
        ranks = torch.argsort(logits_per_audio.flatten(), descending=True).cpu().numpy()
        return ranks
    
    # Load Data #
    try:
        if train_args.prefix:
            prefix = train_args.prefix
    except:
        prefix = ""
        
    text_prompts = [json.loads(line)[args.text_key] for line in open(args.test_file).readlines()]
    text_prompts = [prefix + inp for inp in text_prompts]
    
    test_images = [json.loads(line)[args.img_key] for line in open(args.test_file).readlines()]
    test_images = [prefix + inp for inp in test_images]
    
    if args.num_test_instances != - 1:
        text_prompts = text_prompts[:args.num_test_instances]
        test_images = test_images[:args.num_test_instances]
    
    # Generate #
    num_steps, guidance, batch_size, num_samples = args.num_steps, args.guidance, args.batch_size, args.num_samples
    all_outputs = []
        
    for k in tqdm(range(0, len(text_prompts), batch_size)):
        text = text_prompts[k: k+batch_size]
        imgs = test_images[k: k+batch_size]
        img_stack = []
        
        # add loop
        for img in imgs:
            image = Image.open(img)
            image = image.resize((224,224))
            transform = transforms.Compose([transforms.PILToTensor()])
            img_tensor = transform(image)
            img_tensor = img_tensor.float()
            img_tensor = img_tensor.unsqueeze(0)
            if img_tensor.shape[1] == 1:
                img_tensor = img_tensor.repeat(1,3,1,1)
            img_stack.append(img_tensor)
            
        img_stack = torch.stack(img_stack,dim=0).to(device)
        #print("type of element getting passed ", type(img_stack))
        print("image stack length is ",len(img_stack))
        #print("shape is" , img_stack.shape)
        
        
        with torch.no_grad():
            latents = model.inference(text, img_stack, scheduler, num_steps, guidance, num_samples, disable_progress=True, inference_choice = inference_choice) # changes required here. pass imgs with text or pass the combined feature??
            
            # print("latents shape ", latents.shape)
            # import pdb; pdb.set_trace()
            
            mel = vae.decode_first_stage(latents)

            # import pdb; pdb.set_trace()

            wave = vae.decode_to_waveform(mel)

            # import pdb; pdb.set_trace()
            
            
            all_outputs += [item for item in wave]
            output_dir = output_folder_path
            os.makedirs(output_dir, exist_ok=True)
            for j, wav in enumerate(wave):
                sf.write("{}/output_{}.wav".format(output_dir, j), wav, samplerate=16000)
            # import pdb; pdb.set_trace()
            #print(len(all_outputs), len(wave))
            #print(wave[0].shape)
            #print(abcd)
            
    # Save #
    exp_id = str(int(time.time()))
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)
    
    if num_samples == 1:
        output_dir = "outputs/{}_{}_steps_{}_guidance_{}".format(exp_id, "_".join(args.model.split("/")[1:-1]), num_steps, guidance)
        os.makedirs(output_dir, exist_ok=True)
        for j, wav in enumerate(all_outputs):
            sf.write("{}/output_{}.wav".format(output_dir, j), wav, samplerate=16000)

        result = evaluator.main(output_dir, args.test_references)
        result["Steps"] = num_steps
        result["Guidance Scale"] = guidance
        result["Test Instances"] = len(text_prompts)
        wandb.log(result)
        
        result["scheduler_config"] = dict(scheduler.config)
        result["args"] = dict(vars(args))
        result["output_dir"] = output_dir

        with open("outputs/summary.jsonl", "a") as f:
            f.write(json.dumps(result) + "\n\n")
            
    else:
        for i in range(num_samples):
            output_dir = "outputs/{}_{}_steps_{}_guidance_{}/rank_{}".format(exp_id, "_".join(args.model.split("/")[1:-1]), num_steps, guidance, i+1)
            os.makedirs(output_dir, exist_ok=True)
        
        groups = list(chunks(all_outputs, num_samples))
        for k in tqdm(range(len(groups))):
            wavs_for_text = groups[k]
            rank = audio_text_matching(wavs_for_text, text_prompts[k])
            ranked_wavs_for_text = [wavs_for_text[r] for r in rank]
            
            for i, wav in enumerate(ranked_wavs_for_text):
                output_dir = "outputs/{}_{}_steps_{}_guidance_{}/rank_{}".format(exp_id, "_".join(args.model.split("/")[1:-1]), num_steps, guidance, i+1)
                sf.write("{}/output_{}.wav".format(output_dir, k), wav, samplerate=16000)
            
        # Compute results for each rank #
        for i in range(num_samples):
            output_dir = "outputs/{}_{}_steps_{}_guidance_{}/rank_{}".format(exp_id, "_".join(args.model.split("/")[1:-1]), num_steps, guidance, i+1)
            result = evaluator.main(output_dir, args.test_references)
            result["Steps"] = num_steps
            result["Guidance Scale"] = guidance
            result["Instances"] = len(text_prompts)
            result["clap_rank"] = i+1
            
            wb_result = copy.deepcopy(result)
            wb_result = {"{}_rank{}".format(k, i+1): v for k, v in wb_result.items()}
            wandb.log(wb_result)
            
            result["scheduler_config"] = dict(scheduler.config)
            result["args"] = dict(vars(args))
            result["output_dir"] = output_dir

            with open("outputs/summary.jsonl", "a") as f:
                f.write(json.dumps(result) + "\n\n")
        
if __name__ == "__main__":
    main()
