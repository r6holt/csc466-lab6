# Ryan Holt (ryholt)
# Grayson Clendenon (gclenden)
# Adam Havstad (ahavstad)

import pandas as pd

'''
The sub-matrix including only columns 5, 7, 8, 13, 15, 16, 17,
18, 19, 20 is dense. Almost all users have rated those jokes
'''

def parse(path):
	data = pd.read_csv(path, header=None)

	print(data)
	return data

parse('data/jester-data-1.csv')