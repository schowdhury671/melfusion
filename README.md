# MeLFusion: Synthesizing Music from Image and Language Cues using Diffusion Models, CVPR 2024

## Resources

<a href="https://schowdhury671.github.io/melfusion_cvpr2024/"> 🌐 Webpage </a> | <a href="https://umd0-my.sharepoint.com/:f:/g/personal/sanjoyc_umd_edu/Eok6RG9QIZhNlGubG8-VsDIBhNMK6OOVAWuHpryEC3VnJw"> 🗂️ Datasets </a>

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

The codebase for this work is built on the <a href="https://github.com/declare-lab/tango">Tango </a> and <a href="https://github.com/haoheliu/AudioLDM">AudioLDM </a> repositories. We would like to thank the respective authors for their contribution.

## :mortar_board: Citing MeLFusion

```
@InProceedings{Chowdhury_2024_CVPR,
    author    = {Chowdhury, Sanjoy and Nag, Sayan and Joseph, K J and Srinivasan, Balaji Vasan and Manocha, Dinesh},
    title     = {MeLFusion: Synthesizing Music from Image and Language Cues using Diffusion Models},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {26826-26835}
}
```
