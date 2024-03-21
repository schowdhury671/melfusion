# MeLFusion: Synthesizing Music from Image and Language Cues using Diffusion Models, CVPR 2024

## Proposed Architecture
![alt text](https://github.com/schowdhury671/melfusion/blob/main/diagrams/melfusion_architecture.png)


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


## To run training execute the following with suitable hyperparameters:
```
bash train_mmgen.sh
```

## To run inference:
```
bash inference_mmgen.sh
```
