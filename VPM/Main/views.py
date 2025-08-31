import imghdr
import os
import io
import requests
import torch
import numpy as np
import pandas as pd
from PIL import Image

from django.core.files.storage import FileSystemStorage
from django.conf import settings
from django.shortcuts import render, HttpResponse, redirect
from django.urls import reverse
from torchvision import models, transforms

from Main.models import Feedback

# Precompute + Load resources

# Paths
FEATURES_PATH = os.path.join(settings.BASE_DIR, 'sample_features.npy')
IDS_PATH = os.path.join(settings.BASE_DIR, 'sample_ids.npy')
STYLES_PATH = os.path.join(settings.BASE_DIR, 'sample_styles.csv')

# Load DB features + IDs
features_db = np.load(FEATURES_PATH)
ids_db = np.load(IDS_PATH)

# Load product metadata (name, category, gender, color)
styles_df = pd.read_csv(STYLES_PATH)
styles_map = styles_df.set_index('id').to_dict(orient='index')

# Load pretrained ResNet50 (remove classification layer)
resnet = models.resnet50(pretrained=True)
resnet.eval()
model = torch.nn.Sequential(*list(resnet.children())[:-1])

# Standard preprocessing pipeline for input images
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


# Views

def upload(request):
    """Handles both uploading a file or image URL and Redirecting to search with the uploaded image"""
    image_url, error = None, None

    if request.method == 'POST':
        action = request.POST.get('action')
        uploaded_file = request.FILES.get('image_file')
        entered_url = request.POST.get('image_url', '').strip()

        #Upload Action
        if action == 'upload':
            if uploaded_file:
                # Verify image type
                file_type = imghdr.what(uploaded_file)
                if file_type:
                    fs = FileSystemStorage(location=settings.MEDIA_ROOT, base_url=settings.MEDIA_URL)
                    filename = fs.save(uploaded_file.name, uploaded_file)
                    image_url = fs.url(filename)
                else:
                    error = "Only image files are allowed!"

            elif entered_url:
                # Validate image URL
                try:
                    response = requests.head(entered_url, timeout=5)
                    if response.headers.get('Content-Type', '').startswith('image/'):
                        image_url = entered_url
                    else:
                        error = "The URL does not point to a valid image."
                except Exception:
                    error = "Could not load image from URL."

            else:
                error = "Please upload a file or enter an image URL."

            return render(request, 'upload.html', {'image': image_url, 'error': error})

        #Search Action
        elif action == 'search':
            image_url = request.POST.get('uploaded_image_url', '').strip()
            if not image_url:
                return render(request, 'upload.html', {'error': "Please upload an image first."})

            search_url = reverse('find_similar') + f"?image_url={image_url}"
            return redirect(search_url)

    return render(request, 'upload.html')


def find_similar(request):
    """Takes an image (local or remote), extracts features, compares with DB features, and returns top similar products."""
    image_url = request.GET.get('image_url')
    if not image_url:
        return HttpResponse("No image provided.")

    #Load query image
    try:
        if image_url.startswith('/media/'):
            img_path = os.path.join(settings.MEDIA_ROOT, image_url.replace('/media/', ''))
            img = Image.open(img_path).convert('RGB')
        else:
            response = requests.get(image_url, timeout=10)
            if not response.headers.get('Content-Type', '').startswith('image/'):
                return HttpResponse("The provided URL is not an image.")
            img = Image.open(io.BytesIO(response.content)).convert('RGB')
    except Exception as e:
        return HttpResponse(f"Failed to load image: {e}")

    #Extract features
    img_t = transform(img).unsqueeze(0)
    with torch.no_grad():
        feat = model(img_t).squeeze().numpy()

    #Compute similarities
    similarities = cosine_similarities(feat, features_db)
    top_k = 20
    top_indices = np.argsort(similarities)[::-1][:top_k]

    #Prepare results
    results = []
    for idx in top_indices:
        image_id = int(ids_db[idx])
        sim_score = similarities[idx]

        # Metadata lookup
        meta = styles_map.get(image_id, {})
        results.append((sim_score, {
            'id': image_id,
            'image_url': f"/media/sample_images/{image_id}.jpg",
            'name': meta.get('productDisplayName', f"Product {image_id}"),
            'category': meta.get('masterCategory', 'Unknown'),
            'gender': meta.get('gender', 'Unknown'),
            'base_colour': meta.get('baseColour', 'Unknown'),
        }))

    # Extract filters
    genders = sorted(set([p.get('gender', 'Unknown') for _, p in results]))
    colours = sorted(set([p.get('base_colour', 'Unknown') for _, p in results]))

    return render(request, 'results.html', {
        'query_image': image_url,
        'results': results,
        'genders': genders,
        'colours': colours,
    })


def about(request):
    """Static About page"""
    return render(request, "about.html")


def contact(request):
    """Contact form: saves feedback to DB and returns success flag."""
    context = {}
    if request.method == "POST":
        name = request.POST.get('name')
        email = request.POST.get('email')
        desc = request.POST.get('desc')

        Feedback.objects.create(name=name, email=email, desc=desc)
        context = {"success": True, "name": name}

    return render(request, 'contact.html', context)


# Utilities

def cosine_similarities(query_vec, db_vectors):
    """Compute cosine similarity between a query vector and a DB of vectors."""
    query_vec = query_vec / np.linalg.norm(query_vec)
    db_vectors_norm = db_vectors / np.linalg.norm(db_vectors, axis=1, keepdims=True)
    return np.dot(db_vectors_norm, query_vec)
