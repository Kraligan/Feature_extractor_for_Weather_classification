import os
import pickle

from PIL import Image
from img2vec_pytorch import Img2Vec
from tqdm import tqdm

def extract_features(data_path:str) -> dict:
    features = []
    labels = []
    img2vec = Img2Vec(model='resnet18', cuda=False)

    for label in os.listdir(data_path):
        label_path = os.path.join(data_path, label)

        if not os.path.isdir(label_path):
            continue  # Ignore files like .DS_Store

        for img_name in tqdm(os.listdir(label_path), desc=f"Processing {label}"):
            img_path = os.path.join(label_path, img_name)
            
            try:
                img = Image.open(img_path).convert('RGB')  # convert in case image is RGBA or grayscale
                vec = img2vec.get_vec(img)
                features.append(vec)
                labels.append(label)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

    return features, labels

data_path = 'Data'
features, labels = extract_features(data_path)

print("Sample label:", labels[0])
print("Vector shape:", len(features[0]))

with open("Data/features.pkl", "wb") as f:
    pickle.dump({"features": features, "labels": labels}, f)