class ClinicalTrial:
	"""Object Clinical Trial"""

	def __init__(self, ID, sponsor, published, state, url, ongoing, title, condition, intervention, locations, last_changed, min_age, max_age, genders, health):
		self.id = ID 
		self.sponsor = sponsor
		self.published = published
		self.state = state
		self.url = url
		self.ongoing = ongoing
		self.title = title
		self.condition = condition
		self.intervention = intervention
		self.locations = locations
		self.last_changed = last_changed
		self.min_age = min_age
		self.max_age = max_age
		self.genders = genders
		self.health = health