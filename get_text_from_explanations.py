import sys
import csv
import heapq
import gzip

"""
Extract document texts and labels from explanations file. Usage: <input tsv.gz file> <output tsv file>.
"""

j=0

with gzip.open(sys.argv[1], 'rt', encoding='utf-8') as csvfile:
    with open(sys.argv[2],'w', encoding='utf-8') as outfile:
        reader = csv.reader(csvfile, delimiter='\t')
        reader.__next__()
        last_doc = None
        doc_words = []
        doc_preds = []
        for i,row in enumerate(reader):
            _, doc, true_labels, pred_label, word, attr, posterior = row
            if word == '<s>' or pred_label == 'None' or int(pred_label) > 7:
                    continue
            pred_label = int(pred_label)
            true_labels = true_labels[1:-1].split()

            if i%100000==0:
                print('%s:%d -> %s:%d' % (sys.argv[1], i, sys.argv[2], j))

            if last_doc != doc:
                if last_doc is None:
                    last_doc = doc
                    doc_preds = [pred_label]
                else:
                    true_labels = [int(label) for label in true_labels if int(label) <= 7] ## Skip sub-registers

                    print('\t'.join([doc, str(true_labels), str(doc_preds), ' '.join(doc_words)]), file=outfile)

                    doc_words = []
                    doc_preds = [pred_label]

            if pred_label != doc_preds[-1]:
                doc_preds.append(pred_label)

            if len(doc_preds) < 2: # Do not duplicate words in case of multiple predictions per doc
                doc_words.append(word.lower().replace('</s>',''))

            last_doc = doc
            j += 1

        if not (word == '<s>' or pred_label == 'None' or int(pred_label) > 7):
            true_labels = [int(label) for label in true_labels if int(label) <= 7]
        print('\t'.join([doc, str(true_labels), str(doc_preds), ' '.join(doc_words)]), file=outfile)
