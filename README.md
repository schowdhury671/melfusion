# MelFusion

```
git clone https://github.com/declare-lab/tango/
cd tango

Python version 3.11 (while creating conda env): conda create --name melfusion_env python=3.11

conda activate melfusion_env

clone this repository and go to thecorresponsing folder and execute the following commands: 

pip install -r requirements1.txt
cd diffusers
pip install -e .
pip install -r requirements2.txt

sudo apt-get install lsof
sudo apt install git-lfs
git lfs install

go to cache direction download the following: 
cd ~/.cache   
mkdir audioldm
cd audioldm
wget https://huggingface.co/haoheliu/AudioLDM-S-Full/resolve/main/audioldm-s-full
mv audioldm-s-full audioldm-s-full.ckpt
sudo apt-get install tmux
```
