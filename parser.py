# Ryan Holt (ryholt)
# Grayson Clendenon (gclenden)
# Adam Havstad (ahavstad)

import pandas as pd

'''
The sub-matrix including only columns 5, 7, 8, 13, 15, 16, 17,
18, 19, 20 is dense. Almost all users have rated those jokes
'''

class Parse:
	def __init__(self, path='data/jester-data-1.csv'):
		self.path = path
		self.ratings = None
		self.num_answered = None
		self.joke_avgs = []
		self.user_avgs = []
		self.jokes = 0
		self.users = 0

		self.parse()

	def parse(self):
		data = pd.read_csv(self.path, header=None)
		self.num_answered = data[0].tolist()
		self.ratings = data.drop(0, axis=1)
		self.jokes = self.ratings.shape[1]
		self.users = self.ratings.shape[0]

		for i, row in self.ratings.iterrows():
			total = 0.0
			for v in row.tolist():
				total += v if v != 99 else 0.0
			self.user_avgs.append(round(total/self.jokes, 2))

		for i, col in self.ratings.iteritems():
			total = 0.0
			for v in col.tolist():
				total += v if v != 99 else 0.0
			self.joke_avgs.append(round(total/self.users, 2))

	def print(self):
		print("------------------------PARSED DATA-----------------------")
		print("jokes: {}".format(self.jokes))
		print("users: {}".format(self.users))
		print("\nRATINGS MATRIX: ")
		print(self.ratings.head())
		print("\nNumber of jokes answered per user: ")
		print(self.num_answered[:100])

			


p = Parse('data/jester-data-1.csv')
p.print()