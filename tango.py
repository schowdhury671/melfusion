import json
import torch
from tqdm import tqdm
from huggingface_hub import snapshot_download
from models import AudioDiffusion, DDPMScheduler
from audioldm.audio.stft import TacotronSTFT
from audioldm.variational_autoencoder import AutoencoderKL

class Tango:
    def __init__(self, name="declare-lab/tango", device="cuda:0"):  
        
        path = snapshot_download(repo_id=name, cache_dir="/fs/nexus-scratch/sanjoyc/.cache")
        
        
        #name = "declare-lab/tango-full-ft-audio-music-caps"
        #print("path is ", path)
        #print(abcd)
        
        vae_config = json.load(open("{}/vae_config.json".format(path)))
        stft_config = json.load(open("{}/stft_config.json".format(path)))
        main_config = json.load(open("{}/main_config.json".format(path)))
        
        # print("***********************main_config is ", main_config)
        # print(abcd)
        
        self.vae = AutoencoderKL(**vae_config).to(device)
        self.stft = TacotronSTFT(**stft_config).to(device)
        self.model = AudioDiffusion(**main_config).to(device)
        
        vae_weights = torch.load("{}/pytorch_model_vae.bin".format(path), map_location=device)
        stft_weights = torch.load("{}/pytorch_model_stft.bin".format(path), map_location=device)
        
        
        # main_weights = torch.load("{}/pytorch_model_2.bin".format(path), map_location=device)
        
        # main_weights = torch.load("/sensei-fs/tenants/Sensei-AdobeResearchTeam/share-sanjoyc/tango/finetuned_saved_model_musiccaps_img_run3/best/pytorch_model_2.bin", map_location=device)
        
        main_weights = torch.load("23_aug_mmgen_train_only_alpha_snr5/best/pytorch_model_2.bin", map_location=device)
        
        
        self.vae.load_state_dict(vae_weights)
        self.stft.load_state_dict(stft_weights)
        
        # import pdb; pdb.set_trace()
        self.model.load_state_dict(main_weights, strict=False)

        print ("Successfully loaded checkpoint from:", name)
        
        self.vae.eval()
        self.stft.eval()
        self.model.eval()
        
        self.scheduler = DDPMScheduler.from_pretrained(main_config["scheduler_name"], subfolder="scheduler")
        
    def chunks(self, lst, n):
        """ Yield successive n-sized chunks from a list. """
        for i in range(0, len(lst), n):
            yield lst[i:i + n]
        
    def generate(self, prompt, steps=100, guidance=3, samples=1, disable_progress=True):
        """ Genrate audio for a single prompt string. """
        with torch.no_grad():
            latents = self.model.inference([prompt], self.scheduler, steps, guidance, samples, disable_progress=disable_progress)
            mel = self.vae.decode_first_stage(latents)
            wave = self.vae.decode_to_waveform(mel)
        return wave[0]
    
    def generate_for_batch(self, prompts, steps=100, guidance=3, samples=1, batch_size=8, disable_progress=True):
        """ Genrate audio for a list of prompt strings. """
        outputs = []
        for k in tqdm(range(0, len(prompts), batch_size)):
            batch = prompts[k: k+batch_size]
            with torch.no_grad():
                latents = self.model.inference(batch, self.scheduler, steps, guidance, samples, disable_progress=disable_progress)
                print("@@@@@@@@@@@@@@@@ latents.shape: ", latents.shape)
                mel = self.vae.decode_first_stage(latents)
                print("@@@@@@@@@@@@@@@@ mel.shape: ", mel.shape)
                wave = self.vae.decode_to_waveform(mel)
                print("@@@@@@@@@@@@@@@@ wave.shape: ", wave.shape)
                outputs += [item for item in wave]
        if samples == 1:
            return outputs
        else:
            return list(self.chunks(outputs, samples))
