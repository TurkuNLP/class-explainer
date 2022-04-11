import sys
import collections
import json

""" Evaluate predictive performance based on text files (explanation w-files). """

cntr = collections.defaultdict(lambda: collections.defaultdict(lambda: 0))
line_cnt = 0

for filename in sys.argv[1:]:
	for line in open(filename, encoding='utf-8'):
		fields = line.strip().split('\t')
		trues = json.loads(fields[1])
		preds = json.loads(fields[2])
		for label in trues:
			if label in preds:
				cntr[label]['TP'] += 1
			else:
				cntr[label]['FN'] += 1
		for label in preds:
			if label in trues:
				continue
			cntr[label]['FP'] += 1

print("Label\tPrec\tRecall")
for label in cntr:
	try:
		print(label, cntr[label]['TP']/(cntr[label]['TP']+cntr[label]['FP']), cntr[label]['TP']/(cntr[label]['TP']+cntr[label]['FN']))
	except ZeroDivisionError:
		pass



