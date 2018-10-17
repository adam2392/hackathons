from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect
from django.template import Template, Context
from django.conf import settings
from django.core import serializers
from django.template.loader import get_template
from .tables import ClinicalTrialTable

from .models import Contact
from .forms import ContactForm, SearchForm

import stateQuery
import json, sys, cgi, os

from clinicalsearch.models import ClinicalTrial

# Create your views here.
def index(request):

	return render(request, 'clinicalsearch/index.html', {'form': SearchForm()})

def about(request):

	return render(request, 'clinicalsearch/about.html')

def problem(request):
	return render(request, 'clinicalsearch/problem.html')

def contact(request):
	form_class = ContactForm()

	# check if submit button was pushed
	if request.method == 'POST':
		form = ContactForm(request.POST)

		# check human validity 2+3=5?
		check_human = request.POST.get('human', '')

		# check using django's built in validity test
		if form.is_valid() and int(check_human) is 5:
			# get name, email and content of message
			contact_name = request.POST.get('contact_name', '')
			contact_email = request.POST.get('contact_email', '')
			contact_phone = request.POST.get('contact_phone', '')
			disease_content = request.POST.get('content', '')

			# Save fields to the model
			contact = form.save(commit=True)
			contact.contact_name = contact_name
			contact.contact_email = contact_email
			contact.contact_phone = contact_phone
			contact.content = disease_content
			contact.save()

			# redirect to the contact page
			return render(request, 'clinicalsearch/contact.html', {
					'form': form_class,
					'error': 2,
				})
		else: # form is not valid
			return render(request, 'clinicalsearch/contact.html', {
				'form': form_class,
				'error': 1,
			})

	return render(request, 'clinicalsearch/contact.html', {
		'form': form_class,
		'error': 0,
	})

# def graph(request):
# 	main_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'graph1.txt'))
# 	data = open(main_dir).read()
# 	print data

# 	return HttpResponse((data))

def graph1(request):
	return render(request, 'clinicalsearch/graph1.html')
	
# works
def map(request):
	COUNT = 5000
	jsonList = {} 	# json list to store number of trials/state
	states_abbrev = ['AL','AK','AZ','AR','CA','CO','CT','DE','FL','GA','HI','ID','IL','IN','IA',
	'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI', 'MN', 'MS', 'MO',
	'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC', 'ND', 'OH',
	'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VT',
	'VA', 'WA', 'WV', 'WI', 'WY']

	for index in range(0, len(states_abbrev)):
		state = states_abbrev[index] # the state abbreviation
		trials = ClinicalTrial.objects.filter(ongoing=True, state=state)
		numTrials = len(trials)		 # number of trials per state
		# Create the json object to be returned
		jsonList[state] = {"numTrials": numTrials}

	return render(request, 'clinicalsearch/map.html', {'datum': jsonList})

######################### LIST OF API CALLS ###############################
# not yet working

def searchAPI(request):
	print request.method
	if request.method == 'POST': 
		print "YO"# If the form has been submitted...
		form = SearchForm(request.POST) # A form bound to the POST data
		if form.is_valid(): # All validation rules pass
			data = form.cleaned_data
			params = {}
			trials = ClinicalTrial.objects.all()
			if data["disease"] != '':
				params["disease"] = str(data["disease"])
				trials = trials.filter(condition__contains=data["disease"])
			if data["sponsor"] != '':
				params["sponsor"] = str(data["sponsor"])
				trials = trials.filter(sponsor=data["sponsor"])
			if "genders" in data:
				params["genders"] = str(data["genders"])
				if data["genders"] == "Male":
					trials = trials.filter(genders="Male")
				if data["genders"] == "Female":
					trials = trials.filter(genders="Female")
			if "health" in data:
				params["health"] = str(data["health"])
				if data["health"] == "Must Be Ill":
					trials = trials.filter(health=False)
			if data["age"] != None:
				params["age"] = int(data["age"])
				trials = trials.filter(min_age__lte=params["age"], max_age__gte=params["age"])
			if data["state"] != '':
				print "here"
				ongoingTable = ClinicalTrialTable(trials.filter(state=data["state"], ongoing=True))
				return render(request, 'clinicalsearch/table.html', {"ongoingTable": ongoingTable})
			else:
				jsonList = searchmap(trials)
				return render(request, 'clinicalsearch/searchmap.html', {'datum': jsonList, 'params': params})
	else:
		form = ContactForm() # An unbound form
		print "why again"

def searchmap(trials):
	jsonList = {} 	# json list to store number of trials/state
	states_abbrev = ['AL','AK','AZ','AR','CA','CO','CT','DE','FL','GA','HI','ID','IL','IN','IA',
	'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI', 'MN', 'MS', 'MO',
	'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC', 'ND', 'OH',
	'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VT',
	'VA', 'WA', 'WV', 'WI', 'WY']

	for index in range(0, len(states_abbrev)):
		state = states_abbrev[index] # the state abbreviation
		numTrials = len(trials.filter(ongoing=True, state=state))
		jsonList[state] = {"numTrials": numTrials}

	return jsonList

def diseaseAPI(request):
	disease = request.GET.get('disease')
	data = ClinicalTrial.objects.filter(condition=disease)

	return HttpResponse(json.dumps(data), content_type="application/json")
# works
def stateAPI(request):
	state = request.GET.get('state')
	closed = ClinicalTrial.objects.filter(state=state, ongoing=False)
	ongoing = ClinicalTrial.objects.filter(state=state, ongoing=True)
	data = {"closed": len(closed), "ongoing": len(ongoing)}
	print data
	return HttpResponse(json.dumps(data), content_type="application/json")

def statemapAPI(request):
	state = request.GET.get('state')
	print state
	genders = request.GET.get('genders')
	print genders
	sponsor = request.GET.get('sponsor')
	print sponsor
	health = request.GET.get('health')
	print health
	condition = request.GET.get('condition')
	print condition
	age = request.GET.get('age')
	print age
	closed = ClinicalTrial.objects.filter(state=state, ongoing=False)
	ongoing = ClinicalTrial.objects.filter(state=state, ongoing=True)
	print len(ongoing)
	if condition != None:
		closed = closed.filter(condition__contains=condition)
		ongoing = ongoing.filter(condition__contains=condition)
		print "condition ", len(ongoing)
	if sponsor != None:
		closed = closed.filter(sponsor=sponsor)
		ongoing = ongoing.filter(sponsor=sponsor)
		print "sponsor ", len(ongoing)
	if genders != "Both" and genders != None:
		closed = closed.filter(genders=genders)
		ongoing = ongoing.filter(genders=genders)
		print "genders ", len(ongoing)
	if health == "Must Be Ill":
		closed = closed.filter(health=False)
		ongoing = ongoing.filter(health=False)
		print "health ", len(ongoing)
	if age != None:
		closed = closed.filter(min_age__lte=age, max_age__gte=age)
		ongoing = ongoing.filter(min_age__lte=age, max_age__gte=age)
		print "age ", len(ongoing)

	# closed = ClinicalTrial.objects.filter(state=state, ongoing=False, genders=genders, sponsor=sponsor, health=health, condition__contains=condition, min_age=min_age, max_age=max_age)
	# ongoing = ClinicalTrial.objects.filter(state=state, ongoing=True, genders=genders, sponsor=sponsor, health=health, condition__contains=condition, min_age=min_age, max_age=max_age)
	data = {"closed": len(closed), "ongoing": len(ongoing)}
	print data
	return HttpResponse(json.dumps(data), content_type="application/json")


def completetableAPI(request):
	state = request.GET.get('state')
	data = ClinicalTrial.objects.filter(state=state, ongoing=False)

	genders = request.GET.get('genders')
	print genders
	sponsor = request.GET.get('sponsor')
	print sponsor
	health = request.GET.get('health')
	print health
	condition = request.GET.get('condition')
	print condition
	age = request.GET.get('age')
	print age

	print len(data)
	if condition != None:
		data = data.filter(condition__contains=condition)
		print "condition ", len(data)
	if sponsor != None:
		data = data.filter(sponsor=sponsor)
		print "sponsor ", len(data)
	if genders != "Both" and genders != None:
		data = data.filter(genders=genders)
		print "genders ", len(data)
	if health == "Must Be Ill":
		data = data.filter(health=False)
		print "health ", len(data)
	if age != None:
		data = data.filter(min_age__lte=age, max_age__gte=age)
		print "age ", len(data)

	completeTable = ClinicalTrialTable(data)

	# completeTable = ClinicalTrialTable(ClinicalTrial.objects.filter(state=state, ongoing=False))

	# completeTable = ClinicalTrialTable(ClinicalTrial.objects.filter(state=state, ongoing=False))

	return render(request, 'clinicalsearch/table.html', {"completeTable": completeTable})

def ongoingtableAPI(request):
	state = request.GET.get('state')

	data = ClinicalTrial.objects.filter(state=state, ongoing=True)

	genders = request.GET.get('genders')
	print genders
	sponsor = request.GET.get('sponsor')
	print sponsor
	health = request.GET.get('health')
	print health
	condition = request.GET.get('condition')
	print condition
	age = request.GET.get('age')
	print age

	print len(data)
	if condition != None:
		data = data.filter(condition__contains=condition)
		print "condition ", len(data)
	if sponsor != None:
		data = data.filter(sponsor=sponsor)
		print "sponsor ", len(data)
	if genders != "Both" and genders != None:
		data = data.filter(genders=genders)
		print "genders ", len(data)
	if health == "Must Be Ill":
		data = data.filter(health=False)
		print "health ", len(data)
	if age != None:
		data = data.filter(min_age__lte=age, max_age__gte=age)
		print "age ", len(data)
	
	ongoingTable = ClinicalTrialTable(data)
	return render(request, 'clinicalsearch/table.html', {"ongoingTable": ongoingTable})

def minAgeAPI(request):
	min_age = request.GET.get('min_age')
	data = ClinicalTrial.objects.filter(min_age=min_age)
	print data
	print type(data)
	return HttpResponse(serializers.serialize('json', data), content_type="application/json")

def maxAgeAPI(request):
	max_age = request.GET.get('max_age')
	data = ClinicalTrial.objects.filter(max_age=max_age)
	return HttpResponse(serializers.serialize('json', data), content_type="application/json")

def genderAPI(request):
	genderes = request.GET.get('genders')
	data = ClinicalTrial.objects.filter(genders=genders)
	return HttpResponse(serializers.serialize('json', data), content_type="application/json")
	
def healthAPI(request):
	health = request.GET.get('health')
	data = ClinicalTrial.objects.filter(health=health)
	return HttpResponse(serializers.serialize('json', data), content_type="application/json")
