import sys
import os
import json
import numpy as np
import collections

f1s = collections.defaultdict(lambda: [])
sup = collections.defaultdict(lambda: [])
for filename in sorted(os.listdir(sys.argv[1])):
    if not filename.endswith('.eval.json') or not filename.startswith(sys.argv[2]):
        continue

    data = json.load(open(os.path.join(sys.argv[1], filename)))
    for cl in data:
        if cl.islower() and not cl.endswith('avg'):
            continue
        f1s[cl].append(data[cl]['f1-score'])
        sup[cl].append(data[cl]['support'])

means = []
sds = []
sups = []
print("Class\tF1 M.\tSd.\tSup. M.")
for mean, cl in sorted([(np.mean(f1s[cl]), cl) for cl in f1s if not cl.endswith('avg') and cl.isupper()], key=lambda x:-x[0]):
    print("%s\t%.4f\t%.4f\t%.1f" % (cl, mean, np.std(f1s[cl]), np.mean(sup[cl])))
    means.append(mean)
    sds.append(np.std(f1s[cl]))
    sups.append(np.mean(sup[cl]))

print("%s\t%.4f\t%.4f\t%.1f" % ("Avg.", np.mean(means), np.mean(sds), np.mean(sups)))

cl = 'micro avg'
print("%s\t%.4f\t%.4f\t%.1f" % (cl, np.mean(f1s[cl]), np.std(f1s[cl]), np.mean(sup[cl])))
print("Non-zero runs")
nonzeros = [x for x in f1s[cl] if x > 0]
print("%s\t%.4f\t%.4f\t%.1f" % (cl, np.mean(nonzeros), np.std(nonzeros), len(nonzeros)))
