import sys
import csv

"""
Filter and simplify raw explanations file for further processing. Arguments: <input tsv file> <output tsv file>.
"""

j=0
with open(sys.argv[1], encoding='utf-8') as csvfile:
	with open(sys.argv[2],'w', encoding='utf-8') as outfile:
		reader = csv.reader(csvfile, delimiter='\t')
		reader.__next__()
		for i,row in enumerate(reader):
			_, doc, true_labels, pred_label, word, attr, posterior = row
			if word == '<s>' or pred_label == 'None' or float(attr) < 0:
				continue
			true_labels = true_labels[1:-1].split()
			if pred_label not in true_labels:
				continue
			if i%1000==0:
				print('%s:%d -> %s:%d' % (sys.argv[1], i, sys.argv[2], j))
			print('\t'.join(['doc'+doc[8:], pred_label, word, attr]), file=outfile)
			j += 1
