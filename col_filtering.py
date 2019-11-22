def main():
	path = 'data/jester-data-1.csv'
	# parse tags
	if len(sys.argv) >= 3:
		uid = int(sys.argv[1])
		iid = int(sys.argv[2])
		if(len(sys.argv) == 4):
			method = int(sys.argv[3])
	
	else:
		print("Usage: python3 col_filtering <UserId> <ItemId> [method]")
		return

	p = Parse(path)


"recommendations will be in the range -10 to +10."
"A score of >= 5 is considered a recommended joke."