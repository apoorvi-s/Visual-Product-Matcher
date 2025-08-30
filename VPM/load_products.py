import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from django.core.files import File
import django
import pickle

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "VPM.settings")
django.setup()

from Main.models import Product

model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')

df = pd.read_csv("sample_styles.csv")

for _, row in df.iterrows():
    img_id = str(row['id'])
    img_path = os.path.join("sample_images", f"{img_id}.jpg")

    if not os.path.exists(img_path):
        continue

    img = keras_image.load_img(img_path, target_size=(224, 224))
    x = keras_image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    features = model.predict(x, verbose=0)[0]
    features_binary = pickle.dumps(features)

    product, created = Product.objects.get_or_create(product_id=img_id)

    product.name = row['productDisplayName'] if 'productDisplayName' in row else ''
    product.category = row['subCategory'] if 'subCategory' in row else ''
    product.gender = row['gender'] if 'gender' in row else ''
    product.base_colour = row['baseColour'] if 'baseColour' in row else ''
    product.feature_vector = features_binary

    with open(img_path, 'rb') as f:
        product.image.save(f"{img_id}.jpg", File(f), save=False)

    product.save()
