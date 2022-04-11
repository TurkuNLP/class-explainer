from sklearn.feature_extraction.text import TfidfVectorizer
import collections
import numpy as np
import heapq
import random

PATH = "../../veronika/simplified-data/orig_simplified/en/"
N_KEYWORDS = 100
WORDS_PER_DOC = 20
SAVE_N = 5000

data = collections.defaultdict(lambda: [])
#data = collections.defaultdict(lambda: "")
word_cntr = collections.Counter()

print("Reading data...", flush=True)
for fn in ['train.tsv-simp.tsv', 'dev.tsv-simp.tsv']:
    for row in open(PATH+'/'+fn, encoding="utf-8"):
        labels, text = row.split('\t')
        text = text.strip()
        labels = [l for l in labels.split() if l.isupper()]
        for l in labels:
            try:
                words = random.sample(text.lower().split(), 1000)
            except ValueError:
                words = text.lower().split()
            word_cntr.update(words)
            #data[l] += ' '.join(words)+'\n'
            data[l].append(words)

for label in data:
    print("Filtering infrequent words for", label, flush=True)
    for doc_i, doc in enumerate(data[label]):
        #data[label][doc_i] = ' '.join([w for w in doc if word_cntr[w] >= 5])
        data[label][doc_i] = ' '.join(doc)
    data[label] = '\n'.join(data[label])


def tfidf_keywords_label_as_doc(data):
    print("Fitting TF-IDF model...", flush=True)
    tfidf_vectorizer = TfidfVectorizer(min_df=1, use_idf=True, sublinear_tf=True, max_df=0.5, max_features=15000, smooth_idf=True)
    tfidf_matrix = tfidf_vectorizer.fit_transform(data.values())
    features = tfidf_vectorizer.get_feature_names()

    keywords = {}
    for i, label in enumerate(data):
        print("\nLabel %s, top terms by TF-IDF" % label)
        keywords[label] = []
        for term, score in sorted(list(zip(features,tfidf_matrix.toarray()[i])), key=lambda x:-x[1])[:N_KEYWORDS]:
            print("%.2f\t%s" % (score, term))
            keywords[label].append((term,score))

    return keywords



def tfidf_keywords_top_words_per_doc(data):
    keywords = {}
    for label in data:
        keywords[label] = []
        print("Fitting TF-IDF model for", label)
        tfidf_vectorizer = TfidfVectorizer(min_df=10, use_idf=True, binary=True, max_df=0.5, max_features=15000)
        tfidf_matrix = tfidf_vectorizer.fit_transform(data[label].split('\n'))
        features = tfidf_vectorizer.get_feature_names()
        #key_columns = [i for s,i in heapq.nlargest(SAVE_N, zip(np.asarray(tfidf_matrix.sum(axis=0))[0], range(tfidf_matrix.shape[1])))]
        #new_matrix = tfidf_matrix.toarray()[:,key_columns]
        #new_features = list(np.array(features)[key_columns])
        #new_matrix = tfidf_matrix.toarray()
        new_matrix = np.zeros(tfidf_matrix.shape)
        new_features = features
        #new_matrix = np.zeros(tfidf_matrix.shape)
        for i in range(tfidf_matrix.shape[0]):
            if i % 200 == 0:
                print(".", end="", flush=True)
            for v,j in heapq.nlargest(WORDS_PER_DOC, zip(tfidf_matrix[i].toarray()[0], range(tfidf_matrix.shape[1]))):
                new_matrix[i,j] = v

        print()
        keywords[label] = [(new_features[i],s) for s,i in heapq.nlargest(N_KEYWORDS, zip(new_matrix.sum(axis=0), range(new_matrix.shape[0])))]
        print(' '.join([w for w,_ in keywords[label]]))

        print()
        return keywords

def tfidf_keywords_x(data):
    print("Fitting TF-IDF model...", flush=True)
    docs = [x.split('\n') for x in data.values()]
    group_sizes = [len(x) for x in docs]
    docs = [item for sublist in docs for item in sublist] # Flatten
    tfidf_vectorizer = TfidfVectorizer(min_df=10, use_idf=True, sublinear_tf=True, max_df=np.max(group_sizes)/sum(group_sizes)*0.5, max_features=50000)
    tfidf_matrix = tfidf_vectorizer.fit_transform(docs)
    print("Number of docs, words:", tfidf_matrix.shape)
    features = tfidf_vectorizer.get_feature_names()
    beg = 0
    keywords = {}
    for i, label in enumerate(data):
        print("Label:", label)
        end = beg + group_sizes[i]
        keywords[label] = [(features[i],s) for s,i in heapq.nlargest(2*N_KEYWORDS, zip(np.asarray(tfidf_matrix[beg:end,:].mean(axis=0))[0], range(tfidf_matrix.shape[1])))]
        keywords[label] = [kw for kw in keywords[label] if any([ch for ch in kw if type(ch) is str and ch.isalpha()])][:N_KEYWORDS] # Require letters in token
        print(' '.join([w for w,_ in keywords[label]]))
        beg = end
        print()

    return keywords



out_fn = "eval_output/tfidf_keywords"
keywords = tfidf_keywords_x(data) #tfidf_keywords_top_words_per_doc(data)
for label in keywords:
    with open("%s_%s.tsv" % (out_fn, label), 'w') as kws_out:
        print("\ttoken\tscore_mean\tscore_std\tsource_number", file=kws_out)
        for i, (w, s) in enumerate(keywords[label]):
            print("%d\t%s\t%.5f\t0.0\t1" % (i, w, s), file=kws_out)


"""
for doc_i in range(5):
    print("\nDocument %d, top terms by TF-IDF" % doc_i)
    for term, score in sorted(list(zip(features,tfidf_matrix.toarray()[doc_i])), key=lambda x:-x[1])[:5]:
        print("%.2f\t%s" % (score, term))
"""
