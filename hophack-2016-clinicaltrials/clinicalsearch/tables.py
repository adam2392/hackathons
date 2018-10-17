from models import ClinicalTrial
from table import Table
from table.columns import Column

class ClinicalTrialTable(Table):
	id = Column(field='id', header=u'NCT ID')
	sponsor = Column(field='sponsor', header=u'Sponsor')
	url = Column(field='url', header=u'Link')
	title = Column(field='title', header=u'Study Title')
	condition = Column(field='condition', header=u'Condition')
	intervention = Column(field='intervention', header=u'Intervention')
	date = Column(field='last_changed', header=u'Last Changed')
	ranking = Column(field='ranking', header=u'Rank')

	class Meta:
		model = ClinicalTrial
