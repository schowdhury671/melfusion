# MeLFusion: Synthesizing Music from Image and Language Cues using Diffusion Models, CVPR 2024

## Resources

<a href="https://schowdhury671.github.io/melfusion_cvpr2024/"> 🌐 Webpage </a> | <a href="https://umd0-my.sharepoint.com/personal/sanjoyc_umd_edu/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fsanjoyc%5Fumd%5Fedu%2FDocuments%2Fmelbench&ga=1"> 🗂️ Datasets </a>

## 💡Proposed Architecture
![alt text](https://github.com/schowdhury671/melfusion/blob/main/diagrams/melfusion_architecture.png)


## 🛠️ Environment Preparation
```

Python version 3.11 (while creating conda env): conda create --name melfusion_env python=3.11

conda activate melfusion_env

clone this repository go to the corresponding folder and execute the following commands: 

pip install -r requirements.txt
cd diffusers
pip install -e .
cd audioldm
wget https://huggingface.co/haoheliu/AudioLDM-S-Full/resolve/main/audioldm-s-full
mv audioldm-s-full audioldm-s-full.ckpt

cd ../..
pip install -r requirements2.txt

sudo apt-get install lsof
sudo apt install git-lfs
git lfs install

go to cache and download the following: 
cd ~/.cache   
mkdir audioldm
cd audioldm
wget https://huggingface.co/haoheliu/AudioLDM-S-Full/resolve/main/audioldm-s-full
mv audioldm-s-full audioldm-s-full.ckpt
sudo apt-get install tmux
```


## 🔥 To run training:
```
bash train_mmgen.sh
```

## 💊 To run inference:
```
bash inference_mmgen.sh
```

## 📉 Main Results:
![alt text](https://github.com/schowdhury671/melfusion/blob/main/diagrams/melfusion_main_results.png)


## :pray: Acknowledgements

The codebase for this work is built on the <a href="https://github.com/declare-lab/tango">Tango </a>, <a href="https://github.com/haoheliu/AudioLDM">AudioLDM </a> repository. We would like to thank the respective authors for their contribution.

## :mortar_board: Citing MeLFusion

```
@article{chowdhury2023melfusion,
  author    = {Chowdhury, Sanjoy and Nag, Sayan and K J, Joseph and Vasan Srinivasan, Balaji and Manocha, Dinesh},
  title     = {MeLFusion: Synthesizing Music from Image and Language Cues using Diffusion Models},
  journal   = {CVPR},
  year      = {2024}
}
```
