from django.db import models
import uuid

# Create your models here.

class MlModel(models.Model):
    id = models.UUIDField(primary_key=True,default=uuid.uuid4,editable=False)
    title = models.CharField(max_length=200)
    created_on = models.DateTimeField(auto_now_add=True,editable=False)
    dataset = models.CharField(max_length=200)
    features = models.TextField()
    label = models.CharField(max_length=200)
    model_type = models.CharField(max_length=200,default='classification')
    description = models.TextField(blank=True,null=True)
    test_loss = models.FloatField(blank=True,null=True)
    test_accuracy = models.FloatField(blank=True,null=True)
    train_loss = models.FloatField(blank=True,null=True)
    train_accuracy = models.FloatField(blank=True,null=True)
    train_mean_absolute_error = models.FloatField(blank=True,null=True)
    test_mean_absolute_error = models.FloatField(blank=True,null=True)
    
    def set_features(self,x):
        self.features = "-|-".join(x)
        
    def get_features(self):
        return self.features.split('-|-')
    
    def __str__(self):
        return self.title
