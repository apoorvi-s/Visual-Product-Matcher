from django.db import models

# Create your models here.
from django.db import models

class Product(models.Model):
    image = models.ImageField(upload_to='sample_images/')
    product_id = models.CharField(max_length=50)
    name = models.CharField(max_length=255, blank=True)
    category = models.CharField(max_length=100, blank=True)
    gender = models.CharField(max_length=20, blank=True)
    base_colour = models.CharField(max_length=50, blank=True)
    feature_vector = models.BinaryField()  

    def __str__(self):
        return self.name or self.product_id
