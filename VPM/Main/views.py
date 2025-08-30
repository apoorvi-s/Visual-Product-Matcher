import imghdr
from django.core.files.storage import FileSystemStorage
from django.conf import settings
from django.shortcuts import render, redirect
import requests

def upload(request):
    image_url = None
    error = None

    if request.method == 'POST':
        action = request.POST.get('action')
        uploaded_file = request.FILES.get('image_file')
        entered_url = request.POST.get('image_url', '').strip()

        if action == 'upload':
            if uploaded_file:
                file_type = imghdr.what(uploaded_file)
                if file_type:
                    fs = FileSystemStorage(location=settings.MEDIA_ROOT, base_url=settings.MEDIA_URL)
                    filename = fs.save(uploaded_file.name, uploaded_file)
                    image_url = fs.url(filename)
                else:
                    error = "Only image files are allowed!"
            elif entered_url:
                try:
                    response = requests.head(entered_url, timeout=5)
                    content_type = response.headers.get('Content-Type', '')
                    if content_type.startswith('image/'):
                        image_url = entered_url
                    else:
                        error = "The URL does not point to a valid image."
                except Exception:
                    error = "Could not load image from URL."
            else:
                error = "Please upload a file or enter an image URL."

            return render(request, 'upload.html', {
                'image': image_url,
                'error': error
            })

        elif action == 'search':
            image_url = request.POST.get('uploaded_image_url', '').strip()

            if not image_url:
                error = "Please upload an image first before searching."
                return render(request, 'upload.html', {
                    'image': None,
                    'error': error
                })

            from django.urls import reverse
            from django.http import HttpResponseRedirect

            search_url = reverse('find_similar') + f"?image_url={image_url}"
            return HttpResponseRedirect(search_url)

    return render(request, 'upload.html')
