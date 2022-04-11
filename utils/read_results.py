import json
import numpy as np
import collections
from scipy import stats
import sys

""" Summarize experiment results log. """


data_list = []
data = collections.defaultdict(lambda: [])
for row in open(sys.argv[1]):#"results_log.jsonl"):
    datum = json.loads(row)
    if 'dCoverageNorm' not in datum:
        continue
    data_list.append(('-'.join([str(v) for k,v in datum.items() if k[0] in "sp"]), datum))

data_list.sort(key=lambda x:x[0])

for _,datum in data_list:
    for key in datum:
        if key == 'sData':
            continue
        data[datum['sData']+key].append(datum[key])


for k, v in sorted(data.items(), key=lambda x: len(x)):
    print(k, len(v))

for dataset1 in ['TP']:
    for dataset2 in ['TP']:
        if dataset1 == 'TL' and dataset2 == 'TP':
            continue
        for cov1 in ['cMacroLenNorm', 'cMacroSimple', 'cMicro', 'dKeywords', 'dCoverageNorm']:
            #for cov2 in ['cMacroLenNorm', 'cMacroSimple', 'cMicro', 'dKeywords', 'dCoverageNorm']:
            for cov2 in ['pPredTh', 'pSelFreq', 'pWordsPerDoc']:
                k1 = dataset1+cov1
                k2 = dataset2+cov2
                #if k1 <= k2:# or (dataset1 != dataset2):# and cov1 != cov2):
                #    continue
                try:
                    print("%.4f\t%s\t%s" % (np.corrcoef(data[k1], data[k2])[1,0], k1, k2))
                except:
                    print("Error in corr of", k1, k2)

        print()

print("Mean\tSd\tMin\tMax\tMetric")
for key in data:
    if key[2] not in 'cdp' or key.endswith('Diff') or key.endswith('Ratio') or not data[key]:
        continue
    summary = stats.describe(data[key])
    print("%.4f\t%.4f\t%.4f\t%.4f\t%s" % (summary.mean, np.sqrt(summary.variance), summary.minmax[0], summary.minmax[1], key))

print()
#plot_data = open("results_plot.csv",'w')

def print_results(data_list):
    print("! Filtered results")
    c = 0
    selected = collections.defaultdict(lambda: [])
    print("MiCov\tCov\tCovDist\tKwDist\tPredTh\tSelFreq\tWords/D\tFreqPrTh")
    for key, datum in data_list:
        if datum['sData'] == 'TP':# and datum['dKeywords'] >= 0.7 and datum['dCoverageNorm'] > 0.44 and datum['cMacroSimple'] > 0.05:
            c += 1
            print("%.4f\t%.4f\t%.4f\t%.4f\t%.1f\t%.1f\t%d\t%.1f" % (datum['cMicro'], datum['cMacroSimple'], datum['dCoverageNorm'], datum['dKeywords'], datum['pPredTh'], datum['pSelFreq'], datum['pWordsPerDoc'], datum['pFreqPredTh']))
            for key in datum:
                if key.startswith('p'):
                    selected[key].append(datum[key])

    print("Means:")
    for key in selected:
        print("%.4f\t%s" % (np.mean(selected[key]), key))
    print("%d results" % c)

hm = (lambda x,y: 2*(x*y)/(x+y))
#data_list.sort(key=lambda x: x[1]['cMacroSimple'])
#data_list.sort(key=lambda x: x[1]['dCoverageNorm'])
data_list.sort(key=lambda x: x[1]['cMacroSimple']*x[1]['cMicro'])

print_results(data_list)
