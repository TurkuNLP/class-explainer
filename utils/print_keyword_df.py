import sys
import json
import numpy as np


class_DF = json.load(open("class_df.json"))
class_TF = json.load(open("class_tf.json"))

f = open(sys.argv[1])
f.readline()

label = ''.join([c for c in sys.argv[1] if c.isupper()][-2:])
print("Label:", label)

n_docs = float(class_DF[label]['_N_DOCS'])

def cross_freq(word, lb, freqs):
	cross_keys = [k for k in freqs.keys() if k != lb]
	return np.mean([freqs[l][word]/float(freqs[l]['_N_DOCS']) if word in freqs[l] else 0 for l in cross_keys])

i = 0
DFs = []
TFs = []
distDFs = []
print("\t\tDF\tTF\tDF-dist.")
for row in f:
	nr, token, score, sd, sf = row.split('\t')
	if not any([c.isalpha() for c in token]):
		continue
	i += 1
	df = class_DF[label][token]/n_docs
	DFs.append(df)
	TFs.append(class_TF[label][token]/n_docs)
	bg_df = cross_freq(token, label, class_DF)
	distDFs.append((df-bg_df)/df)
	print('\t'.join([str(x) for x in [token, i, df, class_TF[label][token]/n_docs, distDFs[-1]]]))
	if i>= 100:
		break


print()
print("\t\tDF\tTF\tDF-dist.")
print("Mean\t--\t%.4f\t%.4f\t%.4f" % (np.mean(DFs), np.mean(TFs), np.mean(distDFs)))
print("SD\t--\t%.4f\t%.4f\t%.4f" % (np.std(DFs), np.std(TFs), np.std(distDFs)))
