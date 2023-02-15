import numpy as np
import pandas as pd
import re
import glob
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import sys
import time
from collections import Counter
import collections
#from statistics import multimode
import csv
import json

PATH = ''
NUMBER_OF_KEYWORDS = 100
key_values = ['HI', 'ID', 'IN', 'IP','NA', 'OP']
FREQUENT_PREDICTIONS_THRESHOLD = 0.5


def argparser():
    ap = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    ap.add_argument('--keyword_data', metavar='FILE', required=True,
                    help='Path to keyword data. /* included in call')
    ap.add_argument('--document_data', metavar='FILE', required=True,
                    help='Path to document data. /* included in call')
    ap.add_argument('--number_of_keywords', metavar='INT', type=int, default=NUMBER_OF_KEYWORDS,
                    help='Threshold for number of keywords compared/chosen per register')
    ap.add_argument('--style', metavar='STR', type=str, default='TP',
                    help='TP = True Positive, P = Predictions, TL = True Label')
    ap.add_argument('--prediction_threshold', type=float, required=True,
                    help='Threshold for choosing best predictions from all predcitions')
    ap.add_argument('--frequent_predictions_threshold', type=float, default=FREQUENT_PREDICTIONS_THRESHOLD,
                    help='Threshold for choosing best predictions from all predcitions')
    return ap



############################ PREPROCESSING ################################
def preprocess_text(d):
#    print("prprocess_text l 40", d)
    # Separate punctuations from words by whitespace
    if len(d) == 0:
        d= " "
    d2 = re.sub(r"\s?([^\w\s'/\-\+$]+)\s?", r" \1 ", d)
    return d2.lower()



def preprocess_label(d):
#    print("label to be preprocessed", d)
    #d2 = re.sub(r"\s?([^\w\s'/\-\+$]+)\s?", r" \1 ", d)
    if type(d) is not str:
 #       print("D", d)
        return d
    else:
        d2 = d.replace(',',' ').replace('[','').replace(']','').split()
        new_label = [int(s) for s in d2 if s.isdigit()]
        new_label2 = [s for s in new_label if s < 7]
  #      print("new label2", new_label2)
        return new_label2

def flatten(t):
    """
    Flattens a nested list to a normal list
    [[2,3], 4] = [2,3,4]
    """
    try:
        return [item for sublist in t for item in sublist]
    except TypeError:
        return t

def multimode(data):
    res = []
    data_list1 = Counter(data)
    temp = data_list1.most_common(1)[0][1]
    for ele in data:
        if data.count(ele) == temp:
            res.append(ele)
    res = list(set(res))

    return res

pd.set_option("display.max_rows", None, "display.max_columns", None)

def get_labels(df_list, options):
    """
    Calculates the best predicted label(s) for each doc.
    Also checks, if the prediction(s) is correct.
    Returns new dataframe with one row == one doc
    """


    data = []

    # concatenate all the dataframes
    df = pd.concat(df_list, axis = 0)
    df.to_csv("main_df.csv")
   # print("XX dflist", df_list, flush=True)
    # Loop over all the documents
    for doc in set(df['doc_id']):
       # print("XXX doc", doc, flush=True)
        # get only the ones with this document id, and get the real label
        sub_df = df[df.doc_id == doc]
        #print("sub_df", sub_df)
        true_labels = sub_df['true_label'].iloc[0]
        print("True", true_labels)
        text = sub_df['text'].iloc[0]

        # get all possible predicted labels for this document
        pred_labels = []
        for index, row in sub_df.iterrows():
            #print("row l 110 dis", row)
            pred_labels.append(row['pred_label'])

        pred_labels = flatten(pred_labels)
        print("pred_labels 113 dis", pred_labels)
        # get the most common prediction(s) with multimode, and remove everything over 6
        #most_frequent_predictions_raw = np.array(multimode(pred_labels))
        #most_frequent_predictions = most_frequent_predictions_raw[most_frequent_predictions_raw < 7]

        n_experiments = len(df_list)
        cntr = collections.Counter([x for x in pred_labels if x < 5]) # used to be 7
        most_frequent_predictions = [cl for cl,cnt in cntr.items() if cnt >= n_experiments*options.frequent_predictions_th]



        # if we have (a) suitable prediction(s)
        # see if they are correct or not
        if len(most_frequent_predictions) >0:
            pred_correctly = []
            for lbl in most_frequent_predictions:
                if int(lbl) in true_labels:
                    pred_correctly.append(True)
                else:
                    pred_correctly.append(False)

            # now we have all for saving
            # save doc_id, true_labels, pred_labels, correct predictions, and text
            save_line = [doc, true_labels, most_frequent_predictions, pred_correctly, text ]
            data.append(save_line)
        else:
            # we have no predictions so
            print("Warning: no frequent predictions!")
            pass

    return pd.DataFrame(data = data, columns=['doc_id', 'true_label','pred_label','type','text'])


def filter_n(dfs, n):
    """
    Filters all dataframes in array down to n rows.
    Returns a new array of filtered dataframes.
    """
    df_list = []
    for data in dfs:
        new_data = data.head(n)
        df_list.append(new_data)
    return df_list


def concatenate(dfs, n = 0):
    """
    Concatenates a list of dataframes. If n != 0, also filters them down.
    """
    if n != 0:
        df_list = filter_n(dfs,n)
        return pd.concat(df_list, axis = 1)
    else:
        return pd.concat(dfs, axis=1)





############################## DISTINCTIVENESS ###################################

def get_value_matrix(df):
    """
    for each label's keywords, get the amount of times it is in other label's keywords
    Return the numbers as a matrix
    """

    # saving
    value_list = []

    # for each label, pop out the list and compare to others
    for key in key_values:
        comp = df.pop(key)
        values = []
        # value counts for efficient calculating
        freq = pd.value_counts(df.values.ravel())
        # for all values in current label's keyword list
        for index, row in comp.iteritems():
            try:
                values.append(freq[row])
            except:
                # some values may be null, add None then
                try:
                    if comp.isnull().iloc[index]:
                        values.append(None)
                    else:
                        # if there is no match the word is unique
                        values.append(0)
                except TypeError:
                    values.append(None)
        value_list.append(values)
        # add the column back for next round
        df[key] = comp
    return value_list


def get_0_percentage(values):
    """
    Get the fraction of 0's and other non-null values in a list, the amount of zeros
    and the length of list
    """
    # if there is None, circumvent it by making a new list
    if None in values:
        new_values = []
        for i in values:
            if i != None:
                new_values.append(i)
        values = new_values
    # otherwise just return this
    try:
        return sum(values == np.zeros(len(values)))/ len(values), sum(values == np.zeros(len(values))), len(values)
    except ZeroDivisionError:
        return 0, 0, 0


"""
def distinctiveness(keywords):
    df = keywords
    percent_mean = []
    percent_std = []
    percent_key = []

    print("Taking max ", len(keywords) ,"keywords: (fraction of distinct words, distinct words, all words)")
    # get frequencies
    value_matrix = get_value_matrix(df)
    percent_key = []
    # loop over them and print out info
    for i in range(len(value_matrix)):
        percent_key.append(get_0_percentage(value_matrix[i])[0])
        print(key_values[i], ": ", get_0_percentage(value_matrix[i])[0], " , ", get_0_percentage(value_matrix[i])[1], "/", get_0_percentage(value_matrix[i])[2] )
    percent_mean.append(np.mean(percent_key))
    print("mean: ",np.mean(percent_key))
    percent_std.append(np.std(percent_key))
    print("std dev: ",np.std(percent_key))
    return percent_key, percent_mean, percent_std
"""

def distinctiveness(keywords):
    DF = collections.defaultdict(lambda: 0)
 #   print("XXX keywords l 253 dis", keywords, flush=True)
    for lb in keywords:
    #    print("XXX, lb", lb)
        for kw in keywords[lb].values:
     #       print("XXX kw", kw)
            if type(kw) is not str:
                kw="NaN"
#                kw = kw[0]
            DF[kw] += 1

    dists = []
    for lb in keywords:
        kws=[]
        for kw in keywords[lb].values:
            if type(kw) is not str:
                kw="NaN"
            kws.append(kw)
#        kws = [kw if type(kw) is str else kw="NaN" for kw in keywords[lb].values]

#        kws = [kw if type(kw) is str else kw[0] for kw in keywords[lb].values]
        uniques = sum([1 for kw in kws if DF[kw] == 1])
        dists.append(uniques/len(keywords[lb]))
        print("Distinctivenss of %s: %.2f" % (lb, dists[-1]))

    print("Distictiveness mean:", np.mean(dists))
    return np.mean(dists), np.std(dists)

############################### COVERAGE #################################

def mark_false_predictions(data):
    """
    Make a new column "label" that contains true positives.
    Mark false predictions as None
    """

    labels = []
    for index,row in data.iterrows():
        trues = row['type'].count(True)
        # if all false predictions
        if trues < 1:
            labels.append(None)
        else:
            # drop false predictions (reverse type list do false are dropped)
            labels.append(np.delete(row['pred_label'], [not b for b in row['type']]))

    data['label'] = labels


def process(data, style):
    """
    Keep, as the 'label', the true positives, all predictions, or true labels
    Returns the same data but with added column 'label', which is
    used in the calculations
    """

    if style == "TP":
        # we want to drop all false predictions
        mark_false_predictions(data)
        data.dropna(inplace = True)
        return data

    elif style == "P":
        # we want all the predictions, so pred_label = label
        data['label'] = data['pred_label']
        return data

    elif style == "TL":
        # we want the correct predictions
        data['label'] = data['true_label']
        return data
    else:
        pass


def count_occurrence(keywords, text):
    """
    Count the occurrence of keywords in text
    Text lengths are in number of unique words
    """

    count = 0.0
    assert type(keywords) is not str
    for word in keywords:
        try:
            if word in text.split(" "):   # empty space to remove compound words ect.
                count += 1
        except:   # since there will be null values at the end
            pass
    return count



def get_mean_text_length(data):
    """
    Get the mean text length wrt unique words
    """
    results = [[],[],[],[],[],[],[]]
    for index, row in data.iterrows():
        lbl = row['label']
        for l in lbl:
            results[l].append(row['text_length'])
    means = []
    for i in range(len(results)):
        means.append(np.mean(results[i]))
    return means

def get_mean_text_length_in_char(data):
    """
    Get the mean text length in character number
    """
    results = [[],[],[],[],[],[],[]]
    for index, row in data.iterrows():
        lbl = row['label']
        for l in lbl:
            results[l].append(row['text_length_in_char'])
    means = []
    for i in range(len(results)):
        means.append(np.mean(results[i]))
    return means



def coverage(labelled_predictions,keywords, style):

    # get the column 'label' that is considered the correct label
    # according to the options.style

    df = process(labelled_predictions, style)


    # calculate text lengths (in unique words) for normalizing
    df['text_length'] = df['text'].apply(lambda x: len(np.unique(x.split(" "))))
    df['text_length_in_char'] = df['text'].apply(lambda x: len(x))
    try:
        shortest_doc_len = min(df['text_length'])
    except ValueError:
        shortest_doc_len = 1

    # mean text length, both options
    mean_text_length = get_mean_text_length(df)
    mean_text_length_in_char = get_mean_text_length_in_char(df)

    # theoretical max as 100 (max keywords) * highest max length wrt label / shortest doc overall
    theoretical_max_score = 100*max(mean_text_length)/shortest_doc_len


    # calculate the occurrences in docs, collect data
    l = []
    s = []
    t = []
    for i, row in df.iterrows():
        label_num = row['label']
        # loop over all labels
        for lb in label_num:
            label = key_values[lb]
            text = row['text']
            len_text = len(np.unique(text.split(" ")))
            kw = keywords[label].values
            index = key_values.index(label)

            l.append(label)
            s.append(count_occurrence(kw, text)*(mean_text_length[index] / len_text))
            t.append(row['text'])

    # calculate mean per class
    coverage = pd.DataFrame(data=l, columns= ['label'])
    # divide score with theoretical max for normalisation
    coverage['score'] = np.array(s)/theoretical_max_score
    means = coverage.groupby('label').mean()
    print(means)
    return means.values


def coverage_macro_simple(labelled_predictions,keywords, style):
    # get the column 'label' that is considered the correct label
    # according to the options.style

    df = process(labelled_predictions, style)

    # calculate the occurrences in docs, collect data
    l = []
    s = []
    t = []
    ref = []
    cross_cov_per_class = collections.defaultdict(lambda: [])
 #   print("XXX row label", row['label'])
  #  print("XXX key_values", key_values)
    for i, row in df.iterrows():
        label_num = row['label']
        text = row['text']
        # loop over all labels
        for lb in label_num:
            label = key_values[lb]
            other_labels = [l for i,l in enumerate(key_values) if i not in label_num] # Exclude classes part of hybrids #:lb]+key_values[lb+1:]
            kw = keywords[label].values
            other_kws = []
            ref.append(np.mean([count_occurrence(keywords[other_lb].values, text) / len(keywords[other_lb].values) for other_lb in other_labels]))

            for other_lb in other_labels:
                c = count_occurrence(keywords[other_lb].values, text)
                cross_cov_per_class[other_lb].append(c / len(keywords[other_lb].values))

            index = key_values.index(label)

            l.append(label)
            s.append(count_occurrence(kw, text) / len(kw))
            t.append(row['text'])

    # calculate mean per class
    coverage = pd.DataFrame(data=l, columns= ['label'])
    coverage['score'] = np.array(s)
    means = coverage.groupby('label').mean()
    print(means)
    print("Mean:", np.mean(means.values))

    cross_coverage = pd.DataFrame(data=l, columns= ['label'])
    cross_coverage['score'] = np.array(ref)
    cross_means = cross_coverage.groupby('label').mean()
    print("Cross-coverage:")
    print(cross_means)
    print("Mean:", np.mean(cross_means.values))

    print("Cross-coverage 2:")
    for lb in cross_cov_per_class:
        print(lb, np.mean(cross_cov_per_class[lb]))
    xcov = np.mean([np.mean(x) for x in cross_cov_per_class.values()])
    print("Mean:", xcov)

    #distinctiveness_diff = np.mean(means.values)-np.mean(cross_means.values)
    distinctiveness_diff = np.mean(means.values)-xcov
    #distinctiveness_ratio = np.mean(means.values)/np.mean(cross_means.values)
    distinctiveness_ratio = np.mean(means.values)/xcov
    distinctiveness_norm = distinctiveness_diff/np.mean(means.values)
    print("Coverage-based distinctiveness (delta):", distinctiveness_diff)
    #print("Coverage-based distinctiveness (ratio):", distinctiveness_ratio)
    print("Coverage-based distinctiveness (normalized):", distinctiveness_norm)
    return means.values, distinctiveness_diff, distinctiveness_ratio, distinctiveness_norm


def coverage_micro_multilabel(labelled_predictions,keywords, style):

    # get the column 'label' that is considered the correct label
    # according to the options.style

    df = process(labelled_predictions, style)

    # calculate the occurrences in docs, collect data
    l = []
    s = []
    t = []
    for i, row in df.iterrows():
        label_num = row['label']
        # loop over all labels
        kw_matches = 0
        text = row['text']
        for lb in label_num:
            label = key_values[lb]
            kw = keywords[label].values
            index = key_values.index(label)
            kw_matches += count_occurrence(kw, text)

        try:
            kw_matches /= 100.*len(label_num) # Normalize by max number of kws for all predicted classes
        except ZeroDivisionError:
            kw_matches = 0

        #l.append(label)
        s.append(kw_matches)
        t.append(row['text'])

    """word_coverage = collections.defaultdict(lambda: collections.defaultdict(lambda: 0.))
    for lb in keywords:
        for k_i, kw in enumerate(keywords[lb].values):
            matches = []
            for i, row in df.iterrows():
                text = row['text']
                matches.append(count_occurrence([kw], text))
            word_coverage[kw][lb] = np.mean(matches)
    for lb in keywords:
        for k_i, kw in enumerate(keywords[lb].values):
            cov = word_coverage[kw][lb]
            bg_cov = np.mean([word_coverage[kw][l] for l in keywords])
            dist = (cov-bg_cov)/cov
            print("%s\t%d\t%s\t%.6f" % (lb, k_i+1, kw, dist))
    """

    mean = np.mean(s)
    print('Multilabel coverage mean:', mean)
    return mean


def corpus_coverage(keywords, labelled_predictions, style):
    data = process(labelled_predictions, style)

    data_collection_scores = []
    for key in key_values:
        key_num = key_values.index(key)
        words = []
        scores = []
        for word in keywords[key].dropna():
            word_count = 0.0
            doc_count = 0.0
            for index, row in data.iterrows():
                label_num = row['label']
                for lbl_num in label_num:
                    lbl = key_values[lbl_num]
                    txt = row['text']
                    if lbl == key:
                            doc_count += 1
                            if word in txt:
                                word_count += 1
            #print("word: ", word)
            #print("in this many docs: ", word_count)
            #print("all docs with label: ", doc_count)
            words.append(word)
            if doc_count > 0:
                scores.append(word_count/doc_count)
            else:
                scores.append(np.nan)
        #filename = "roskaa/Testttiiiuwu"+key+".csv"
        #with open(filename, "w") as f:
        #    for i in range(len(words)):
        #        line = str(words[i])+ ","+str(scores[i])+"\n"
        #        f.write(line)
        #f.close()
        print(key, ": ", np.mean(scores), " support: ", doc_count)
        data_collection_scores.append(np.mean(scores))

    return data_collection_scores


def coverage_df(keywords, labelled_predictions, style):
    data = process(labelled_predictions, style)
    cntr = collections.defaultdict(lambda: collections.defaultdict(lambda: 0))

    for index, row in data.iterrows():
        label_nums = row['label']
        for lbl_num in label_nums:
            lbl = key_values[lbl_num]
            txt = row['text']
            for word in set(txt.split()):
                cntr[lbl][word] += 1
            cntr[lbl]['_N_DOCS'] += 1

    data_collection_scores = []
    for lbl in key_values:
        #key_num = key_values.index(lbl)
        score = 0

        for keyword in keywords[lbl].dropna():
            score += cntr[lbl][keyword]

        score /= cntr[lbl]['_N_DOCS']

        print(lbl, ": ", score, " support: ", cntr[lbl]['_N_DOCS'])
        data_collection_scores.append(score)

    print("Mean:", np.mean(data_collection_scores))
    return data_collection_scores


def corpus_coverage3(keywords):
    import json
    DF = json.load(open("class_df.json"))
    data_collection_scores = []
    for key in key_values:
        scores = []
        key_num = key_values.index(key)
        for word in keywords[key].dropna():
            try:
                scores.append(DF[key][word])
            except KeyError:
                scores.append(0)

        scores_mean = np.mean(scores)/DF[key]['_N_DOCS']

        print(key, ": ", scores_mean, " support: ", DF[key]['_N_DOCS'])
        data_collection_scores.append(scores_mean)

    return data_collection_scores



def evaluate_keywords_file(options):
    data_list = []
    num_files = 0
    print("eval fires")
#    try:
 #       labelled_docs = pd.read_csv("%s-w_agg-%.1f.tsv" % (options.document_data, options.frequent_predictions_th),sep="\t",converters={'type': lambda x: [bool(y) for y in x.strip("[]").split(", ")], 'true_label': json.loads, 'pred_label': json.loads})
  #  except:# FileNotFoundError: # modif 2906, 

#/run-fi-2-wNF.tsv

    for filename in glob.glob(options.document_data+"*wNF.tsv"):
            #try:
        print("filename l 645 dis", filename)
        num_files +=1
#            print(filename, flush = True)
        raw_data = pd.read_csv(filename, sep='\t', names=['doc_id', 'true_label', 'pred_label', 'text'])
            # remove null predictions
        raw_data.dropna(axis = 0, how='any', inplace = True)
            # add white space to punctuation and lowercase the letters
        raw_data['text'] = raw_data['text'].apply(preprocess_text)
            # add commas to multilabels and change them  to numeric data
        raw_data['true_label'] = raw_data['true_label'].apply(preprocess_label)
        raw_data['pred_label'] = raw_data['pred_label'].apply(preprocess_label)
            # add a tag for the source file
        raw_data['source'] = filename
        data_list.append(raw_data)
            #except:
            #    print("Error at ", filename, flush=True)

    print("XXXXXX l 662 dis Length of data_list:", len(data_list), flush=True)
#    print("XXX data list", data_list, flush=True)
        # get the most common prediction for all documents
        # prediction must be 0...6
        # if there are labels between 0 and 6 that are equally good
        # keep them both
    labelled_docs = get_labels(data_list, options)

    print("XXX l 670labeled docs head", labelled_docs.head(30), flush=True)
    labelled_docs.to_csv("%s-w_agg-%.1f.tsv" % (options.document_data, options.frequent_predictions_th), sep='\t')

    #labelled_docs = pd.read_csv(options.document_data+'w_agg.tsv', sep='\t')

    labels = "HI ID IN IP NA OP".split()
  #  print("XXX kwlist")
 #   try:
 #       print("token l 681", labels)#(options.keyword_data+l+'.tsv', sep='\t')[['token']].rename(columns={"token": l}) for l in labels)
 #   except:
 #       print("fail l 683 d")
    kw_list = [pd.read_csv(options.keyword_data+l+'.tsv', sep='\t')[['token']].rename(columns={"token": l}) for l in labels]
#    print("l 682 d file", options.keyword_data+l+'.tsv')
    print("XXX line 677 kwlist", kw_list)
    keywords = concatenate(kw_list, options.number_of_keywords)

    # NOW all is preprocessed
    print("l 687 d Preprocessing done", flush=True)

    # THE CALCULATIONS

    print("Distictiveness of keywords: ", flush=True)
    dist_mean, dist_std = distinctiveness(keywords=keywords)


    for style in ['TP', 'TL']:
        print("Data subset:", style)
        #print("Coverage: ", flush=True)
        #coverage_scores_macro_lennorm = coverage(labelled_docs, keywords, style)
        print("Coverage (macro simple): ", flush=True)
        try:
            coverage_scores_macro_simple, cov_dist_diff, cov_dist_ratio, cov_dist_norm = coverage_macro_simple(labelled_docs, keywords, style)
        except:
            print("coverage failed")
        print("Coverage (micro multilabel): ", flush=True)
        coverage_scores_micro = coverage_micro_multilabel(labelled_docs, keywords, style)
        #print("Coverage (DF): ", flush=True)
        #coverages4 = coverage_df(keywords, labelled_docs, options.style)

        with open("results_log_ext.jsonl",'a') as logfile:
            print("Logging results to results_log_ext.jsonl")
            log_output = {'sData': style,\
                          #'cMacroLenNorm': np.mean(coverage_scores_macro_lennorm),\
                          'cMacroSimple': np.mean(coverage_scores_macro_simple),\
                          'cMicro': coverage_scores_micro,\
                          'dKeywords': dist_mean,\
                          #'dCoverageDiff': cov_dist_diff,\
                          #'dCoverageRatio': cov_dist_ratio,\
                          'dCoverageNorm': cov_dist_norm,\
                          'pFreqPredTh': options.frequent_predictions_th}
            logfile.write(json.dumps(log_output)+'\n')




def calculate(options):


    data_list = []
    num_files = 0

    try:
        labelled_docs = pd.read_csv("%s-w_agg-%.1f.tsv" % (options.document_data, options.frequent_predictions_th),sep="\t",converters={'type': lambda x: [bool(y) for y in x.strip("[]").split(", ")], 'true_label': json.loads, 'pred_label': json.loads})
    except FileNotFoundError:
        print("XXX line 734 d")
        for filename in glob.glob(options.document_data+"*w.tsv"):
            #try:
            num_files +=1
            print(filename, flush = True)
            raw_data = pd.read_csv(filename, sep='\t', names=['doc_id', 'true_label', 'pred_label', 'text'])
            # remove null predictions
            raw_data.dropna(axis = 0, how='any', inplace = True)
            # add white space to punctuation and lowercase the letters
            raw_data['text'] = raw_data['text'].apply(preprocess_text)
            # add commas to multilabels and change them  to numeric data
            raw_data['true_label'] = raw_data['true_label'].apply(preprocess_label)
            raw_data['pred_label'] = raw_data['pred_label'].apply(preprocess_label)
            # add a tag for the source file
            raw_data['source'] = filename
            data_list.append(raw_data)
            #except:
            #    print("Error at ", filename, flush=True)

        print("Length of data_list:", len(data_list))

        # get the most common prediction for all documents
        # prediction must be 0...6
        # if there are labels between 0 and 6 that are equally good
        # keep them both
        labelled_docs = get_labels(data_list, options)

        #print(labelled_docs.head(30))
        labelled_docs.to_csv("%s-w_agg-%.1f.tsv" % (options.document_data, options.frequent_predictions_th), sep='\t')

    #labelled_docs = pd.read_csv(options.document_data+'w_agg.tsv', sep='\t')

    # read the keywords per class
    HI = pd.read_csv(options.keyword_data+'HI.tsv', sep='\t')[['token']].rename(columns={"token": "HI"})
    IN = pd.read_csv(options.keyword_data+'IN.tsv', sep='\t')[['token']].rename(columns={"token": "IN"})
    ID = pd.read_csv(options.keyword_data+'ID.tsv', sep='\t')[['token']].rename(columns={"token": "ID"})
    IP = pd.read_csv(options.keyword_data+'IP.tsv', sep='\t')[['token']].rename(columns={"token": "IP"})
   # df_LY = pd.read_csv(options.keyword_data+'LY.tsv', sep='\t')[['token']].rename(columns={"token": "LY"})
    NA = pd.read_csv(options.keyword_data+'NA.tsv', sep='\t')[['token']].rename(columns={"token": "NA"})
    OP = pd.read_csv(options.keyword_data+'OP.tsv', sep='\t')[['token']].rename(columns={"token": "OP"})

    kw_list = [HI, ID, IN, IP, NA, OP]
    keywords = concatenate(kw_list, options.number_of_keywords)

    # NOW all is preprocessed
    print("Preprocessing done l 773 dis", flush=True)




    print("Parameters:")
    print("  Prediction threshold:", options.prediction_th)
    print("  Selection frequency:", options.selection_freq)
    print("  Words per document:", options.words_per_doc)
    print("  Frequent predictions threshold:", options.frequent_predictions_th)
    # THE CALCULATIONS

    print("Distictiveness of keywords: ", flush=True)
    dist_mean, dist_std = distinctiveness(keywords=keywords)
    for style in ['TP', 'TL']:
        print("Data subset:", style)
        #print("Coverage: ", flush=True)
        #coverage_scores_macro_lennorm = coverage(labelled_docs, keywords, style)
        print("Coverage (macro simple): ", flush=True)
        coverage_scores_macro_simple, cov_dist_diff, cov_dist_ratio, cov_dist_norm = coverage_macro_simple(labelled_docs, keywords, style)
        print("Coverage (micro multilabel): ", flush=True)
        coverage_scores_micro = coverage_micro_multilabel(labelled_docs, keywords, style)
        #print("Coverage (DF): ", flush=True)
        #coverages4 = coverage_df(keywords, labelled_docs, options.style)

        with open("results_log.jsonl",'a') as logfile:
            print("Logging results to results_log.jsonl")
            log_output = {'sData': style,\
                          #'cMacroLenNorm': np.mean(coverage_scores_macro_lennorm),\
                          'cMacroSimple': np.mean(coverage_scores_macro_simple),\
                          'cMicro': coverage_scores_micro,\
                          'dKeywords': dist_mean,\
                          #'dCoverageDiff': cov_dist_diff,\
                          #'dCoverageRatio': cov_dist_ratio,\
                          'dCoverageNorm': cov_dist_norm,\
                          'pPredTh': options.prediction_th,\
                          'pSelFreq': options.selection_freq,\
                          'pWordsPerDoc': options.words_per_doc,\
                          'pFreqPredTh': options.frequent_predictions_th}
            logfile.write(json.dumps(log_output)+'\n')


    """
    # change to float
    save_data = np.array([dist_data_per_label,flatten(coverages),flatten(coverages2),coverages3]).astype('float')

    # make new dataframe
    df_save = pd.DataFrame(data = key_values, columns=['label'])
    df_save['distinctiveness'] = save_data[0]
    df_save['coverage'] = save_data[1]
    df_save['corpus_macro'] = save_data[2]
    df_save['corpus_micro'] = save_data[3]
    df_save['delta'] = df_save['distinctiveness'].values.astype('float')- df_save['coverage'].values.astype('float')


    # add the mean of each column
    mean_of_values = df_save.mean(axis = 0).values
    std_of_values = df_save.std(axis=0).values
    mean_row = flatten(['-', mean_of_values])
    std_row = flatten(['-', std_of_values])
    new_row= {'label':'Mean', 'distinctiveness':mean_row[1], 'coverage':mean_row[2], 'corpus_coverage':mean_row[3], 'delta': mean_row[4]}
    new_row2 = {'label':'Std', 'distinctiveness':std_row[1], 'coverage':std_row[2], 'corpus_coverage':std_row[3], 'delta': std_row[4]}
    df_save_0 = df_save.append(new_row, ignore_index=True)
    df_save_new = df_save_0.append(new_row2, ignore_index=True)

    # display
    print(df_save_new, flush=True)

    # saving? just df_save.to_csv(filename, sep='\t')
    """

if __name__=="__main__":
    options = argparser().parse_args(sys.argv[1:])
    #calculate(options)
    #evaluate_keywords_file(options)
