import torch
from torchvision import models, transforms
from PIL import Image
import os
import pandas as pd
import numpy as np

# Paths — update if needed
images_dir = r'D:\Third Year Projects\Visual-Product-Matcher\VPM\sample_images'
csv_path = r'D:\Third Year Projects\Visual-Product-Matcher\VPM\sample_styles.csv'
features_out_path = r'D:\Third Year Projects\Visual-Product-Matcher\VPM\sample_features.npy'
ids_out_path = r'D:\Third Year Projects\Visual-Product-Matcher\VPM\sample_ids.npy'

# Load CSV
df = pd.read_csv(csv_path)

# Load pretrained ResNet50 model
model = models.resnet50(pretrained=True)
model.eval()

# Remove last classification layer to get features
feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])

# Define image transforms
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
])

features = []
image_ids = []

for idx, row in df.iterrows():
    img_id = row['id']
    img_path = os.path.join(images_dir, f"{img_id}.jpg")

    if not os.path.exists(img_path):
        print(f"Image missing: {img_path}")
        continue

    img = Image.open(img_path).convert('RGB')
    img_t = transform(img).unsqueeze(0)

    with torch.no_grad():
        feat = feature_extractor(img_t)  # Output shape: (1, 2048, 1, 1)

    feat = feat.squeeze().numpy()  # shape (2048,)

    features.append(feat)
    image_ids.append(img_id)

# Save features and IDs as numpy arrays
np.save(features_out_path, np.stack(features))
np.save(ids_out_path, np.array(image_ids))

print(f"Extracted features for {len(features)} images and saved to disk.")
