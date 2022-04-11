import sys
import collections
import numpy as np
from sklearn.metrics import classification_report

""" Evaluate predictive performance based on prediction files (p-files) """


PATH = "explanations/rerun%.3dp.tsv"

f1s = collections.defaultdict(lambda: [])
sups = collections.defaultdict(lambda: [])

for run in range(100):
	true_mx = []
	pred_mx = []
	print(PATH % run)
	f = open(PATH % run, encoding='utf-8')
	f.__next__()

	confmx = collections.defaultdict(lambda: collections.defaultdict(lambda: 0))
	last_doc = None
	last_trues = []
	preds = []
	for row in f:
		try:
			_, doc, trues, pred, _ = row.split('\t')
		except:
			continue
		if pred == 'None':
			pred = None
		else:
			pred = int(pred)
		if last_doc != doc:
			"""for i in range(8):
				if i in last_trues:
					if i in preds:
						confmx[i]['TP'] += 1
					else:
						confmx[i]['FN'] += 1
				else:
					if i in preds:
						confmx[i]['FP'] += 1
					else:
						confmx[i]['TN'] += 1
			"""
			true_mx.append([1 if i in last_trues else 0 for i in range(8)])
			pred_mx.append([1 if i in preds else 0 for i in range(8)])
			preds = []
	
		if pred is not None and pred > 7:
			continue
		preds.append(pred)
		trues = [int(x) for x in trues[1:-1].split()]
		last_trues = [x for x in trues if x <= 7]
		last_doc = doc

	if preds != []:
		true_mx.append([1 if i in last_trues else 0 for i in range(8)])
		pred_mx.append([1 if i in preds else 0 for i in range(8)])

	"""print("label\tprec\trecall\tf1\tsup")
	for i in confmx:
		#print(i, confmx[i]['TP'], confmx[i]['FN'], confmx[i]['FP'], confmx[i]['TN'])
		try:
			prec = confmx[i]['TP']/(confmx[i]['TP']+confmx[i]['FP'])
		except ZeroDivisionError:
			prec = 0.
		recall = confmx[i]['TP']/(confmx[i]['TP']+confmx[i]['FN'])
		try:
			f1 = 2*prec*recall/(prec+recall)
		except ZeroDivisionError:
			f1 = 0.
		sup = confmx[i]['TP']+confmx[i]['FN']
		print("%d\t%.4f\t%.4f\t%.4f\t%d" % (i, prec, recall, f1, sup))
		f1s[i].append(f1)
		sups[i].append(sup)

	TP = sum([x['TP'] for x in confmx.values()])
	FP = sum([x['FP'] for x in confmx.values()])
	FN = sum([x['FN'] for x in confmx.values()])

	try:
		prec = TP/(TP+FP)
	except ZeroDivisionError:
		prec = 0.
	recall = TP/(TP+FN)
	try:
		f1 = 2*prec*recall/(prec+recall)
	except ZeroDivisionError:
		f1 = 0.

	print("micro\t%.4f\t%.4f\t%.4f\t%d" % (prec, recall, f1, TP+FN))

	f1s['micro'].append(f1)
	sups['micro'].append(TP+FN)
	"""
	
	rep = classification_report(np.array(true_mx), np.array(pred_mx), output_dict=True)
	weighted = sum([rep[str(i)]['f1-score']*rep[str(i)]['support'] for i in range(8)])/sum([rep[str(i)]['support'] for i in range(8)])
	print("weighted", weighted, "micro", rep['micro avg']['f1-score'])

	if rep['micro avg'] == 0:
		continue
	for lb in rep:
		f1s[lb].append(rep[lb]['f1-score'])
		sups[lb].append(rep[lb]['support'])


print("\nMean:\tLabel\tF1\tStd\tSup")
for k in f1s:
	print("%s\t%.4f\t%.4f\t%.4f" % (str(k), np.mean(f1s[k]), np.std(f1s[k]), np.mean(sups[k])))

print("Non-zero")
for k in f1s:
	print("%s\t%.4f\t%.4f\t%.4f" % (str(k), np.mean([x for x,y in zip(f1s[k],f1s['micro avg']) if y > 0]), np.std([x for x,y in zip(f1s[k],f1s['micro avg']) if y > 0]), np.mean([x for x,y in zip(sups[k],f1s['micro avg']) if y > 0])))

print("%s\t%.4f\t%.4f\t%.4f" % ('micro', np.mean([x for x in f1s['micro avg'] if x > 0]), np.std([x for x in f1s['micro avg'] if x > 0]), np.mean(sups['micro avg'])))





"""
0       t31930  [ 2 10] 2       "[-2.1090775 -4.8605146  1.9511603 -2.5116663 -6.9758205 -3.819534
 -2.3805091 -5.5522895 -3.4385026 -3.0007606 -0.6639879 -6.4854617
 -4.6914964 -2.9807057 -5.7114935 -4.29253   -6.571276  -5.1867642
 -4.7974157 -4.261508  -4.352073  -5.9613028 -5.888267  -5.1896486

"""
