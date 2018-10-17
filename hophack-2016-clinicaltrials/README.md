# clinicaltrials
Hophacks hackathon with Adam Li, Richard Chen and Daniel Chen.

Background:
This web app was built with a backend of Django, data mining scripts and analysis scripts. The frontend consists of HTML, CSS, Bootstrap and D3.js. We were inspired by the lack of functionality and user experience from clinicaltrials.gov. 

Running Analysis:
Use /clinicalsearch/scraper.py to run web scraper based on the output from the clinicalgovs API. This will gather the corresponding information into trials.csv. Afterwards use the R script in the root directory to create trials_ranked.csv and use populatedb.py to populate the database with the corresponding information. This will setup the entire web app to run correspondingly.

To run entire website, pull from github.
* run virtualenv venv
* source venv/bin/activate
* pip install -r requirements.txt
* go to mysite directory and run 'python manage.py runserver'

admin:
clinicalsearch
health

References:
* http://datamaps.github.io/
* https://clinicaltrials.gov/
* https://pypi.python.org/pypi/clinical_trials/1.1
* https://github.com/shymonk/django-datatable
