import os
import shutil
import pandas as pd

dataset_dir = r'D:\Fashion-product-images'
images_dir = os.path.join(dataset_dir, 'images')
csv_file = os.path.join(dataset_dir, 'styles.csv')
target_dir = r'D:\Third Year Projects\Visual-Product-Matcher\VPM\sample_images'

os.makedirs(target_dir, exist_ok=True)

df = pd.read_csv(csv_file, on_bad_lines='skip') 

df = df.dropna(subset=['id'])

image_ids = df['id'].astype(int).head(100).tolist()

copied_ids = []
for image_id in image_ids:
    filename = f"{image_id}.jpg"
    src = os.path.join(images_dir, filename)
    dst = os.path.join(target_dir, filename)
    if os.path.exists(src):
        shutil.copy(src, dst)
        copied_ids.append(image_id)
    else:
        print(f"File not found: {src}")

# Save filtered CSV
df_small = df[df['id'].isin(copied_ids)]
csv_out_path = r'D:\Third Year Projects\Visual-Product-Matcher\VPM\sample_styles.csv'
df_small.to_csv(csv_out_path, index=False)

print("Copied 100 images and saved sample_styles.csv")
