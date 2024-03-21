import cv2

from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel
# from datasets import load_dataset
from transformers import AutoProcessor, ClapModel
import librosa

# Paths to the images
image_path1 = 'hbd-1.jpg'
image_path2 = 'hbd-2.jpg'

# Read the images
image1 = cv2.imread(image_path1)
image2 = cv2.imread(image_path2)


model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)
image_list = [image1, image2]

inputs = processor(text=["Another year, another reason to celebrate!", "Happy birthday to you. Many many happy returns of the day"], images=image_list, return_tensors="pt", padding=True)

outputs = model(**inputs)
logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
probs_clip = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities


audio_file_path1 = 'hbd1.mp3'
# Load the audio file
y1, sr1 = librosa.load(audio_file_path1, sr=None)

audio_file_path2 = 'hbd2.mp3'
# Load the audio file
y2, sr2 = librosa.load(audio_file_path2, sr=None)

audio_sample = [y1, y2]



model = ClapModel.from_pretrained("laion/clap-htsat-unfused")
processor = AutoProcessor.from_pretrained("laion/clap-htsat-unfused")

input_text = ["Another year, another reason to celebrate!", "Happy birthday to you. Many many happy returns of the day"]

inputs = processor(text=input_text, audios=audio_sample, return_tensors="pt", padding=True)

outputs = model(**inputs)
logits_per_audio = outputs.logits_per_audio  # this is the audio-text similarity score
probs_clap = logits_per_audio.softmax(dim=-1)  # we can take the softmax to get the label probabilities


# imsm_score = probs_clip @ probs_clap.T  # image x audio

probs_metric = probs_clip @ probs_clap.T
imsm_score = probs_metric.softmax(dim = -1)

print("imsm:",imsm_score)
 