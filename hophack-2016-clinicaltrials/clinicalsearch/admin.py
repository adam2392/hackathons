from django.contrib import admin
from .models import Contact
from .models import ClinicalTrial

# Register your models here.
admin.site.register(Contact)
admin.site.register(ClinicalTrial)