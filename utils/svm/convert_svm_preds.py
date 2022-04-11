import csv
import gzip
import collections
#import heapq
import numpy as np
import re
import json
from sklearn.metrics import classification_report


PREDS_FILE = "../../veronika/simplified-data/data_shuffs/predictions/split%s.preds.gz"
WORDS_FILE = "explanations/rerun%swNF.tsv"

FEATS_FILE = "../../veronika/simplified-data/data_shuffs/feats/split%s.feats.gz"

OUT_FILE_WORDS = "svm_expl/rerun%sw.tsv"
OUT_FILE_KEYWORDS = "svm_expl/keywords_%s.tsv"

SELECTION_FREQUNECY = 0.6
N_RUNS = 100

csv.field_size_limit(1000000)

cnt_hybrids = 0

fscores = collections.defaultdict(lambda: [])
supports = collections.defaultdict(lambda: [])
for n in range(0, N_RUNS):
    n = "%.3d" % n

    lookup = {}
    with open(WORDS_FILE % n, 'r', encoding='utf-8') as wordsfile:
        words_reader = csv.reader(wordsfile, delimiter='\t')
        for row in words_reader:
            idx, true, pred, text = row
            lookup[idx] = (true, pred, text)
        print("Read %d docs from words file." % len(lookup))

    out_words = open(OUT_FILE_WORDS % n, 'w')

    preds = []
    trues = []
    print("Converting predictions for %s..." % n)
    with gzip.open(PREDS_FILE % n, 'rt') as predfile:
        pred_reader = csv.reader(predfile, delimiter='\t')
        pred_reader.__next__()
        #probs_buffer = []
        for i, row in enumerate(pred_reader):
            try:
                #probs, pred, true, true_class, idx = row
                pred, true, true_class, idx = row
            except ValueError:
                #probs_buffer.append(' '.join(row))
                print("Error:", row)
                continue
            if true_class == 'OTHER':
                continue
            #probs = ''.join(probs_buffer)+probs
            #probs_buffer = []
            if idx.startswith('train'):
                new_index = 't'
            else:
                new_index = 'd'
            new_index += idx.split('+')[1]

            if not idx.endswith('+0'):
                if int(pred) != preds[-1]:
                    print("HYBRID PRED", new_index, pred, str(preds[-1]))
                    cnt_hybrids += 1
                if int(true) != trues[-1]:
                    #print("HYBRID TRUE", new_index, true, str(trues[-1]))
                    #preds.append(int(pred))
                    #trues.append(int(true))
                    preds[-1][int(pred)] = 1
                    trues[-1][int(true)] = 1
                continue

            try:
                assert new_index in lookup
            except AssertionError:
                print(row)
                continue

            #print('\t'.join([str(i), new_index, true, pred, probs]))
            print('\t'.join([new_index, str([int(true)]), pred]), lookup[new_index][2].encode('utf-8'), file=out_words)
            preds.append(int(pred))
            trues.append(int(true))
            preds.append([1 if int(pred) == i else 0 for i in [0, 1, 2, 3, 4, 5, 6, 8])
            trues.append([1 if int(true) == i else 0 for i in [0, 1, 2, 3, 4, 5, 6, 8])

        trues = np.array(trues)
        preds = np.array(preds)
        print(classification_report(trues, preds))
        rep = classification_report(trues, preds, output_dict=True)
        for lb in rep:
            if lb.isdigit():
               fscores[lb].append(rep[lb]['f1-score'])
               supports[lb].append(rep[lb]['support'])
        try:
            fscores['micro'].append(rep['accuracy'])
        except:
            fscores['micro'].append(rep['micro avg'])

print("Hybrids:", cnt_hybrids)

print("Label\tmean\tstd\tsup")
for lb in fscores:
    if not lb.isdigit():
        continue
    print("%s\t%.4f\t%.4f\t%.4f" % (lb, np.mean(fscores[lb]), np.std(fscores[lb]), np.mean(supports[lb])))

print("%s\t%.4f\t%.4f\t--" % ('micro', np.mean([x['f1-score'] for x in fscores['micro']]), np.std([x['f1-score'] for x in fscores['micro']])))

print("Loading keyword data...")
scores = collections.defaultdict(lambda: collections.defaultdict(lambda: {}))

for n in range(0, N_RUNS):
    n = "%.3d" % n
    print(n, end=" ", flush=True)
    with gzip.open(FEATS_FILE % n, 'rt', encoding='utf-8') as featsfile:
        feats_reader = csv.reader(featsfile, delimiter='\t', quoting=csv.QUOTE_NONE)
        feats_reader.__next__()
        #probs_buffer = []
        for i, row in enumerate(feats_reader):
            _, cl, weight, feat, sign, label = row
            if not any([ch.isalpha() for ch in feat]) or sign != 'pos':
                continue
            feat = re.sub(r"^([\.,:;\!\?\"\(\)])+", r"", feat) # strip punctuation at beginning
            feat = re.sub(r"([\.,:;\!\?\"\(\)])+$", r"", feat) # strip punctuation at end

            scores[label][n][feat.lower()] = float(weight)

"""
print()
print("Cropping lists...")
MAX_INTERMEDIATE_KWS = 1000
N_KEYWORDS = 100
for lb in scores:
    print(lb, end=" ", flush=True)
    for n in scores[lb]:
        for kw in scores[lb][n]:
            scores[lb][n] = dict(heapq.nlargest(MAX_INTERMEDIATE_KWS, scores[lb][n].items(), key=lambda x: x[1]))
"""
print()
scores_ = collections.defaultdict(lambda: collections.defaultdict(lambda: {}))
print("Inverting index...")
for lb in scores:
    print(lb, end=" ", flush=True)
    for n in scores[lb]:
        for kw in scores[lb][n]:
            scores_[lb][kw][n] = scores[lb][n][kw]

print()
print("Selecting stable keywords...")
stable_kws = collections.defaultdict(lambda: {})
stable_kws_sd = collections.defaultdict(lambda: {})
for lb in scores_:
    print(lb, end=" ", flush=True)
    for kw in scores_[lb]:
        if len(scores_[lb][kw])/float(N_RUNS) >= SELECTION_FREQUNECY:
            stable_kws[lb][kw] = np.mean(list(scores_[lb][kw].values()))
            stable_kws_sd[lb][kw] = np.std(list(scores_[lb][kw].values()))


"""class_DFs = json.load(open('class_df.json'))

print()
N_KEYWORDS = 100
print("Keywords:")
for lb in stable_kws:
    kws = []
    with open(OUT_FILE_KEYWORDS % lb, 'w') as outfile_kw:
        print("Class:", lb)
        print("\ttoken\tscore_mean\tscore_std\tsource_number", file=outfile_kw)
        for kw, score in sorted(stable_kws[lb].items(), key=lambda x:-x[1]):
            try:
                if class_DFs[lb][kw] >= 5:
                    kws.append(kw)
                    print("%s\t%f\t%d" % (kw, score, len(scores_[lb][kw])))
                    print("%d\t%s\t%f\t%f\t%d" % (len(kws), kw, score, stable_kws_sd[lb][kw], len(scores_[lb][kw])), file=outfile_kw)
            except:
                pass
            if len(kws) >= N_KEYWORDS:
                break

        outfile_kw.close()
        print()
"""
