import sys
import json
import numpy as np

MIN_WORD_FREQ = 5
SELECTION_FREQUNECY = 60

class_df = json.load(open("class_df.json"))

tokens = []
for fn in sys.argv[1:]:
	print("Reading", fn)
	f = open(fn)
	f.__next__()

	for row in f:
		try:
			_, token, pred, score, std, sel_freq = row.split('\t')
		except:
			_, token, score, std, sel_freq = row.split('\t')

		tokens.append((float(score), int(sel_freq), token))
	print("Tokens:", len(tokens))


tokens.sort()
tokens.reverse()

print("Score\tSF\tRank\tRawrank\tKeyword")
unrank, rank = 0, 0
label = ''.join([ch for ch in sys.argv[1] if ch.isupper()])
SFs = []
for score, sel_freq, token in tokens:
	try:
		if class_df[label][token] >= MIN_WORD_FREQ and any([ch.isalpha() for ch in token]):
			unrank += 1
			if sel_freq >= SELECTION_FREQUNECY:
				rank += 1
				print("%.4f\t%d\t%d\t%d\t%s" % (score, sel_freq, rank, unrank, token))
				SFs.append(sel_freq)
	except KeyError:
		pass
	if rank >= 100:
		break

print("Mean SF:", np.mean(SFs))
