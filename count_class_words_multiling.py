from sklearn.feature_extraction.text import CountVectorizer
import collections
import json

""" Calculate term and document frequencies in corpus and save to file.

File 'class_df.json' is required for metric calculations in evaluation (run_evaluation.py). """


PATH = '../../veronika/simplified-data/en/'
PATH = 'oscar_data'

class_DF = collections.defaultdict(lambda: collections.defaultdict(lambda: 0))
class_TF = collections.defaultdict(lambda: collections.defaultdict(lambda: 0))
vectorizer = CountVectorizer()
for lang in ['ar', 'en', 'fi', 'fr', 'zh']:
    dataset = 'oscar_data_%s_40k.tsv' % lang
    for i, row in enumerate(open(PATH+'/'+lang+'/'+dataset, encoding='utf-8')):
        try:
            labels, text = row.split('\t')
        except:
            continue
        labels = labels.strip().split(' ')
        labels = [l for l in labels if l in set("HI ID IN IP LY NA OP SP".split())]
        #text = ' '.join(text.strip().split(' ')[:300]) # approximation of token context size
        text = ' '.join(text.strip().split(' '))
        try:
            m = vectorizer.fit([text])
            #for word in m.get_feature_names():
            #    for label in labels:
            #        class_DF[label][word] += 1
            for word, cnt in zip(m.get_feature_names(), m.transform([text]).toarray()[0]):
                word = str(word)
                for label in labels:
                    class_TF[label][word] += int(cnt)
                    class_DF[label][word] += 1
        except ValueError:
            continue
        for label in labels:
            class_DF[label]['_N_DOCS'] += 1
        if i % 100 == 0:
            print(' '.join(["%s:%d" % x for x in sorted([(k, len(v)) for k,v in class_DF.items()], key=lambda x:-x[1])]))
            print()#print("N docs:", n_docs)

json.dump(class_DF, open("class_df.json",'w'))
json.dump(class_TF, open("class_tf.json",'w'))
