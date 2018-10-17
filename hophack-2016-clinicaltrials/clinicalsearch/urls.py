from django.conf import settings
from django.conf.urls import url
from . import views

urlpatterns = [
	url(r'^$', views.index, name='index'),
	url(r'^index', views.index, name='index'),
	url(r'^about', views.about, name='about'),
	url(r'^problem', views.problem, name='problem'),
	url(r'^map', views.map, name='map'),
	url(r'^contact', views.contact, name='contact'),
	url(r'^graph1', views.graph1, name='graph1'),

	# Data Processing URLs
	url(r'^api/search', views.searchAPI, name='searchAPI'),
	url(r'^api/statemap', views.statemapAPI, name='statemapAPI'),
	url(r'^api/getstatedata', views.stateAPI, name='stateAPI'),
	url(r'^api/getdiseasedata', views.diseaseAPI, name='diseaseAPI'),

	url(r'^api/completetable', views.completetableAPI, name='completetableAPI'),
	url(r'^api/ongoingtable', views.ongoingtableAPI, name='ongoingtableAPI'),

	url(r'^api/getminagedata', views.minAgeAPI, name='minAgeAPI'),
	url(r'^api/getmaxagedata', views.maxAgeAPI, name='maxAgeAPI'),
	url(r'^api/getgenderdata', views.genderAPI, name='genderAPI'),
	url(r'^api/gethealthdata', views.healthAPI, name='healthAPI'),
]