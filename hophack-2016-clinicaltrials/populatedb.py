# Run this in the Django shell

from clinicalsearch.models import ClinicalTrial
import csv

with open('clinicalsearch/trials_ranked.csv', 'rU') as csvfile:
	reader = csv.reader(csvfile, delimiter=',')
	for row in reader:
		print row
		t = ClinicalTrial(id=row[0], sponsor=row[1], published=(row[2]=="TRUE"), state=row[3], url=row[4], ongoing=(row[5]=="TRUE"), title=row[6], condition=row[7], intervention=row[8], locations=row[9], last_changed=row[10], min_age=int(row[11]), max_age=int(row[12]), genders=row[13], health=(row[14] == "True"), ranking=int(row[15]))
		t.save()