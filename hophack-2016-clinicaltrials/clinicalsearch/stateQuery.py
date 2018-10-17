############# Function to get data from clinicaltrials.gov 
# using clinical api
# from django.conf.settings import STATIC_ROOT
from clinical_trials import Trials
import json, os
from ClinicalTrial import ClinicalTrial
from django.conf import settings

# list of states abbreviations and corresponding states
states_abbrev = ['AL','AK','AZ','AR','CA','CO','CT','DE','FL','GA','HI','ID','IL','IN','IA',
	'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI', 'MN', 'MS', 'MO',
	'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC', 'ND', 'OH',
	'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VT',
	'VA', 'WA', 'WV', 'WI', 'WY']
states = ['Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado', 'Connecticut',
		 'Delaware', 'Florida', 'Georgia', 'Hawaii', 'Idaho', 'Illinois', 'Indiana', 'Iowa',
		 'Kansas', 'Kentucky', 'Louisiana', 'Maine', 'Maryland', 'Massachusetts', 'Michigan',
		 'Minnesota', 'Mississippi', 'Missouri', 'Montana', 'Nebraska', 'Nevada', 'New Hampshire',
		 'New Jersey', 'New Mexico', 'New York', 'North Carolina', 'North Dakota', 'Ohio', 'Oklahoma',
		 'Oregon', 'Pennsylvania', 'Rhode Island', 'South Carolina', 'South Dakota', 'Tennessee', 'Texas',
		 'Utah', 'Vermont', 'Virginia', 'Washington', 'West Virginia', 'Wisconsin', 'Wyoming']

COUNT = 5000 # the number of trials we want to query

# 03: Function to help populate db with clinicaltrial objects
def get_state_trials():
	# trialsList = TRIALS_LIST
	
	# create a clinical trials object for searching
	t = Trials()
	trialsList = []	# list to store all 50 state's active clinical trials

	for index in range(0, len(states_abbrev)):
		trials = t.search(recruiting='open', count=COUNT, state=states_abbrev[index])['search_results']['clinical_study']

		# create a whole number of lists for passing in urls/etc.
		trialsList.append(trials) 

	# clinical trial object list
	clinical_meta_list = []
	# loop through list of state's trials
	for i in range(0, len(trialsList)):
		# Get the corresponding state's trials
		stateTrials = trialsList[i]

		# loop through each state's trials
		for j in range(0, len(stateTrials)):
			# create a holder for this trial
			trial = stateTrials[j]

			if 'intervention_summary' not in trial:
				continue

			# Query the trial ID, and state it is in
			nct_id = trial['nct_id']
			state = states_abbrev[i]
			# sponsor = line
			url = trial['url']
			last_changed = trial['last_changed']
			title = trial['title']
			condition = trial['condition_summary']
			intervention = trial['intervention_summary']

			# Create a ClinicalTrial Object to hold relevant data
			clinical_meta_data = ClinicalTrial(nct_id, None, None, state, url, True, title, condition, intervention, None, last_changed, None, None, None, None)
			# Add object to a list
			clinical_meta_list.append(clinical_meta_data)

	return clinical_meta_list

# Gets closed trials for each state
def get_closed_trials():
	t = Trials()
	clinical_trials = []
	for state in states_abbrev:
		closed_trials = t.search(recruiting='closed', count=COUNT, state=state)['search_results']['clinical_study']
		for trial in closed_trials:
			if 'intervention_summary' not in trial:
				continue
			# Get the trial ID, url
			nct_id = trial['nct_id']
			url = trial['url']
			title = trial['title']
			condition = trial['condition_summary']
			intervention = trial['intervention_summary']
			last_changed = trial['last_changed']

			# Create a ClinicalTrial Object to hold relevant data
			clinical_trial = ClinicalTrial(nct_id, None, None, state, url, False, title, condition, intervention, None, last_changed, None, None, None, None)
			
			# Add object to a list
			clinical_trials.append(clinical_trial)

	return clinical_trials

# # 04: Function to return a list of trials by sponsor (restricted to USA for now)
# def get_sponsor_trials():
# 	# clinical trial object list
# 	clinical_meta_list = []

# 	sponsorList = [] # list of sponsors from txt file

# 	# directory of sponsors.txt file
# 	main_dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))
# 	data_file_path = os.path.join(main_dir, 'sponsors.txt')
		
# 	# create a clinical trials object for searching
# 	t = Trials()

# 	# Open local file to get list of sponsors
# 	with open(data_file_path, 'r') as file:
# 		# loop through each sponsor
# 		for line in file:
# 			trial_search = t.search(country='US', sponsor=line)
# 			if int(trial_search['search_results']['count']) > 0:
# 				trial_results = trial_search['search_results']['clinical_study']

# 				# convert to list if only single object returned
# 				if not isinstance(trial_results, type([])):
# 					trial_results = [trial_results]
# 				# Loop through each trial for a certain sponsor
# 				for i in range(0, len(trial_results)):
# 					trial = trial_results[i]

# 					# Query the trial ID, and state it is in
# 					nct_id = trial['nct_id']
# 					# state = states_abbrev[i]
# 					sponsor = line
# 					url = trial['url']
# 					title = trial['title']
# 					condition = trial['condition_summary']
# 					try:
# 						intervention = trial['intervention_summary']
# 					except:
# 						intervention = None

# 					last_changed = trial['last_changed']

# 					# check if open, or closed
# 					is_closed = trial['status']['open']
# 					if is_closed == "Y":
# 						is_closed = True
# 					elif is_closed == "N":
# 						is_closed = False

# 					# Create a ClinicalTrial Object to hold relevant data
# 					clinical_meta_data = ClinicalTrial(nct_id, None, None, state, url, True, title, condition, intervention, None, last_changed)
# 					# Add object to a list
# 					clinical_meta_list.append(clinical_meta_data)

# 			else: # empty query result
# 				print "Emtpy count for: ", line
# 				print trial_search['search_results']['count']

# 			sponsorList.append(line)

# 	return clinical_meta_list

