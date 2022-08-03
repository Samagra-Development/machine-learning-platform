from rest_framework import serializers
from .models import MlModel


class MlModelSerializer(serializers.ModelSerializer):
    class Meta:
        model = MlModel
        fields = '__all__'