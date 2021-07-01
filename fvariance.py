import sys
import os
import json
import numpy as np
import collections

f1s = collections.defaultdict(lambda: [])
sup = collections.defaultdict(lambda: [])
for filename in sorted(os.listdir(sys.argv[1])):
    if not filename.endswith('.eval.json'):
        continue

    data = json.load(open(os.path.join(sys.argv[1], filename)))
    for cl in data:
        f1s[cl].append(data[cl]['f1-score'])
        sup[cl].append(data[cl]['support'])

print("Class\tF1 M.\tSd.\tSup. M.")
for mean, cl in sorted([(np.mean(f1s[cl]), cl) for cl in f1s], key=lambda x:-x[0]):
    if cl.endswith('avg'):
        continue
    print("%s\t%.4f\t%.4f\t%.1f" % (cl, mean, np.std(f1s[cl]), np.mean(sup[cl])))

cl = 'micro avg'
print("%s\t%.4f\t%.4f\t%.1f" % (cl, np.mean(f1s[cl]), np.std(f1s[cl]), np.mean(sup[cl])))
