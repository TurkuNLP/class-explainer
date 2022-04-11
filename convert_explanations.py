import sys
import csv
import collections
import numpy as np

"""
Filter and simplify raw explanations file for further processing. Arguments: <input tsv prefix (e.g. run000)>.
Reads p-files (document-level predictions) and a-files (word-level attributions) in order to procude s-files 
(aggregated token scores) and w-files ("words", i.e. document texts and labels; wNF = no filtering).
"""

j=0

with open(sys.argv[1]+'p.tsv', encoding='utf-8') as pred_file:
	pred_reader = csv.reader(pred_file, delimiter='\t')
	pred_reader.__next__()
	preds = [(docID, true, pred, pred_vals[1:-1].split()) for _, docID, true, pred, pred_vals in pred_reader]
	pred_dict = collections.defaultdict(lambda: [])
	pred_vals_dict = collections.defaultdict(lambda: [])
	true_dict = {}
	for docID, true, pred, pred_vals in preds:
		pred_dict[docID].append(pred)
		pred_vals_dict[docID].append([float(x) for x in pred_vals])
		true_dict[docID] = true[1:-1].split()

alt_thresholds = [0.7, 0.9]
path = '/'.join(sys.argv[1].split('/')[:-1])
fn_prefix = sys.argv[1].split('/')[-1]

alt_scores_files = [open("%s/th%.1f/%ss.tsv" % (path, th, fn_prefix), 'w', encoding='utf-8') for th in alt_thresholds]


missing_markers = 0
hist = collections.defaultdict(lambda: 0)
with open(sys.argv[1]+'a.tsv', encoding='utf-8') as attr_file:
	with open(sys.argv[1]+'s.tsv', 'w', encoding='utf-8') as scores_file:
		scores_buffer = []
		with open(sys.argv[1]+'wNF.tsv', 'w', encoding='utf-8') as words_file:
			attr_reader = csv.reader(attr_file, delimiter='\t')
			attr_reader.__next__()
			last_doc = None
			doc_words = []
			saved_words = True
			for i, row in enumerate(attr_reader):
				_, docID, word, attr = row

				if word == '</s>':# and int(pred_dict[docID][pred_nr]) < 8:
					print('\t'.join([docID, str([int(x) for x in true_dict[docID]]), pred_dict[docID][pred_nr], ' '.join(doc_words).lower()]), file=words_file)
					saved_words = True

					for line in scores_buffer:
						print(line, file=scores_file)

					logits = [1.0/(1.0 + np.exp(- x)) for x in pred_vals_dict[docID][pred_nr]]
					for th, alt_f in zip(alt_thresholds, alt_scores_files):
						if logits[int(pred_dict[docID][pred_nr])] > th:
							for line in scores_buffer:
								print(line, file=alt_f)

					scores_buffer = []
					j += 1
					continue

				if word == '<s>' or last_doc != docID:
					if not saved_words:
						missing_markers += 1
						print('\t'.join([last_doc, str([int(x) for x in true_dict[last_doc]]), pred_dict[last_doc][pred_nr], ' '.join(doc_words).lower()]), file=words_file)
					saved_words = False
					if last_doc != docID:
						pred_nr = 0
						last_doc = docID
						doc_words = []
						if word != '<s>':
							doc_words.append(word)
					else:
						pred_nr += 1
					continue

				if pred_nr == 0:
					doc_words.append(word)

				if pred_dict[docID] == 'None' or attr == "None" or float(attr) < 0:
					continue

				if int(pred_dict[docID][pred_nr]) >= 8:
					continue

				if pred_dict[docID][pred_nr] not in true_dict[docID]:
					continue

				if i%1000==0:
					print('%s:%d -> %d' % (sys.argv[1], i, j))

				scores_buffer.append('\t'.join([docID, pred_dict[docID][pred_nr], word, attr]))
				j += 1
				hist[pred_nr] += 1


print("%d docs missing </s>" % missing_markers)


for i, c in hist.items():
	print(i,c)
