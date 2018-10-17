from django.db import models

class ClinicalTrial(models.Model):
	id = models.TextField(primary_key=True)
	sponsor = models.TextField()
	published = models.BooleanField()
	state = models.TextField()
	url = models.TextField()
	ongoing = models.BooleanField()
	title = models.TextField()
	condition = models.TextField()
	intervention = models.TextField()
	locations = models.TextField()
	last_changed = models.TextField()
	ranking = models.IntegerField()
	min_age = models.IntegerField()
	max_age = models.IntegerField()
	genders = models.TextField()
	health = models.BooleanField()


class Contact(models.Model):
	id = models.TextField(primary_key=True)
	contact_name = models.CharField(max_length=200)
	contact_phone = models.CharField(max_length=15)
	contact_email = models.CharField(max_length=30)
	content = models.TextField()
