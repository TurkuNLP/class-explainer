import numpy as np
import pandas as pd
import re
import glob
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import sys
import time
from collections import Counter
from statistics import multimode
import csv

PATH = '/content/gdrive/MyDrive/Colab Notebooks/NLP/uniqueness/'
UNIQUENESS_THRESHOLDS = [100]
KEYWORD_THRESHOLD = 0
key_values = ['HI', 'ID', 'IN', 'IP','LY', 'NA', 'OP']



def argparser():
    ap = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    ap.add_argument('--keyword_data', metavar='FILE', required=True,
                    help='Path to keyword data. /* included in call')
    ap.add_argument('--document_data', metavar='FILE', required=True,
                    help='Path to document data. /* included in call')
    ap.add_argument('--keyword_threshold', metavar='INT', type=int, default=KEYWORD_THRESHOLD,
                    help='Threshold for number of keywords compared/chosen per register')
    ap.add_argument('--coverage_threshold', metavar='FLOAT', type=float,
                help='Threshold for the number of correct predictions for a document')
    return ap




def flatten(t):
    """
    Flattens a nested list to a normal list
    [[2,3], 4] = [2,3,4]
    """
    return [item for sublist in t for item in sublist]

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

def get_value_matrix(df):
    value_list = []
    for key in key_values:
        comp = df.pop(key)
        values = []
        freq = pd.value_counts(df.values.ravel())
        for index, row in comp.iteritems():
            try:
                values.append(freq[row])
            except:
                if comp.isnull().iloc[index]:
                    values.append(None)
                else:
                    values.append(0)
        value_list.append(values)
        df[key] = comp
    return value_list


def get_0_percentage(values):
    if None in values:
        new_values = []
        for i in values:
            if i != None:
                new_values.append(i)
        values = new_values
    return sum(values == np.zeros(len(values)))/ len(values), sum(values == np.zeros(len(values))), len(values)




def preprocess_text(d):
    # Separate punctuations from words by whitespace
    d2 = re.sub(r"\s?([^\w\s'/\-\+$]+)\s?", r" \1 ", d)
    return d2.lower()
    


def preprocess_label(d):
    d2 = re.sub(r"\s?([^\w\s'/\-\+$]+)\s?", r" \1 ", d)
    new_label = []
    #past = ""
    #for l in d.split(" "):
    #    if l.isnumeric():
    #        new_label.append(int(l))
    #return new_label
    new_label = [int(s) for s in d2.split() if s.isdigit()]
    return new_label


def count_occurence(keywords, text, index, mean_text_length):

    count = 0.0
    for word in keywords:
        try:
            if " "+word+" " in text:   # empty space to remove compound words ect.
                count += 1
        except:   # since there will be null values at the end
            pass
    return count*(mean_text_length[index] / len(text))




def all_labels(df_list):

    data = []
    df = pd.concat(df_list, axis = 0)
    for doc in set(df['doc_id']):
        sub_df = df[df.doc_id == doc]
        pred_labels = []
        for index, row in sub_df.iterrows():
            pred_labels.append(row['pred_label'])
        
        most_frequent_predictions = multimode(pred_labels)
        most_frequent_predictions.sort()
        pred_correctly = "corpus"
        pred = -1
        for lbl in most_frequent_predictions:
            if int(lbl) in sub_df['true_label'].iloc[0]:
                pred = lbl
                pred_correctly = "predicted"
                break
        if pred == -1:
            pred = most_frequent_predictions[0]
        
        #most_frequent_prediction = max(set(pred_labels), key = pred_labels.count)
        save_line = [doc, sub_df['true_label'].iloc[0],pred, most_frequent_predictions, pred_correctly,sub_df['text'].iloc[0] ]
        data.append(save_line)
    return pd.DataFrame(data = data, columns=['doc_id', 'true_label','pred_label','most_frequent','type','text'])






def uniqueness(keywords):
    df = keywords
    percent = []
    percent_std = []
    percent_key = []

    print("Taking max ", len(keywords) ,"keywords: (% of unique words, unique words, all words)")
    value_matrix = get_value_matrix(df)
    percent_key = []
    for i in range(len(value_matrix)):
        percent_key.append(get_0_percentage(value_matrix[i])[0])
        print(key_values[i], ": ", get_0_percentage(value_matrix[i])[0], "% , ", get_0_percentage(value_matrix[i])[1], "/", get_0_percentage(value_matrix[i])[2] )
    percent.append(np.mean(percent_key))
    print("mean: ",np.mean(percent_key))
    percent_std.append(np.std(percent_key))
    print("std dev: ",np.std(percent_key))


def coverage(labelled_predictions,keywords, thr):
    



    # get only predicted, not corpus
    # comment this if you want to save type for analysis, and uncomment saving below
    df = labelled_predictions[labelled_predictions.type == "predicted"]
    #df = labelled_predictions
    
    # calculate text lengths for normalizing
    df['text_length'] = df['text'].apply(lambda x:len(x))
    
    # mean text length per label
    mean_text_length_s = df.groupby('pred_label')['text_length'].mean()
    
    # change it to a list
    mean_text_length = []
    for row in mean_text_length_s:
        mean_text_length.append(float(row))


    # calculate the occurrences in docs, collect data
    l = []
    s = []
    t = []
    c = []
    for index, row in df.iterrows():
        label_num = row['pred_label']
        label = key_values[label_num]
        text = row['text']
        kw = keywords[label].values
        index = key_values.index(label)
  
        l.append(label)
        s.append(count_occurence(kw, text, index, mean_text_length))
        t.append(row['text'])
        c.append(row['type'])

    # if you want to save, uncomment these, and comment type removal from above
    #save_df = pd.DataFrame(data = l, columns=['label'])
    #save_df['score'] = s
    #save_df['type'] = c
    #save_df.to_csv("coverage_scores2.tsv", sep="\t")
    

    # calculate mean per class
    coverage = pd.DataFrame(data=l, columns= ['label'])
    coverage['score'] = np.array(s)
    coverage['type'] = np.array(c)


    print(coverage['label'].value_counts())
    means = coverage.groupby('label').mean()
    print(means)


def corpus_coverage(keywords, data, all = False):

    
    for key in key_values:
        key_num = key_values.index(key)
        words = []
        scores = []
        for word in keywords[key].dropna():
            word_count = 0.0
            doc_count = 0.0
            for index, row in data.iterrows():
                lbl_num = row['pred_label']
                lbl = key_values[lbl_num]
                txt = row['text']
                tpe = row['type']
                if lbl == key:
                    #if tpe == "predicted":
                        doc_count += 1
                        if word in txt:
                            word_count += 1
            print("word: ", word)
            print("in this many docs: ", word_count)
            print("all docs with label: ", doc_count)
            words.append(word)
            scores.append(word_count/doc_count)
        filename = "corpus_coverage_all_predictions_top100_kw_"+key+".csv"
        with open(filename, "w") as f:
            for i in range(len(words)):
                line = str(words[i])+ ","+str(scores[i])+"\n"
                f.write(line)
        f.close()


    
def corpus_coverage_true_label(keyword, data):
    
    for key in key_values:
        key_num = key_values.index(key)
        words = []
        scores = []
        for word in keywords[key].dropna():
            word_count = 0.0
            doc_count = 0.0
            for index, row in data.iterrows():
                label_num = row['pred_label']
                for lbl_num in label_num:
                    lbl = key_values[lbl_num]
                    txt = row['text']
                    tpe = row['type']
                    if lbl == key:
                        #if tpe == "predicted":
                            doc_count += 1
                            if word in txt:
                                word_count += 1
            print("word: ", word)
            print("in this many docs: ", word_count)
            print("all docs with label: ", doc_count)
            words.append(word)
            scores.append(word_count/doc_count)
        filename = "corpus_coverage_all_predictions_top100_kw_"+key+".csv"
        with open(filename, "w") as f:
            for i in range(len(words)):
                line = str(words[i])+ ","+str(scores[i])+"\n"
                f.write(line)
        f.close() 
        
                




options = argparser().parse_args(sys.argv[1:])
data_list = []
num_files = 0
current_time = time.time()
print(current_time)

# Read all the files
for filename in glob.glob(options.document_data+"/*.tsv"):
    try:
        num_files +=1
        print(filename, flush = True)
        raw_data = pd.read_csv(filename, sep='\t', index_col=0).rename(columns={"0":'doc_id', "1":'true_label', "2":'pred_label',"3":'text'})
        # remove null predictions
        raw_data.dropna(axis = 0, how='any', inplace = True)
        # add white space to punctuation and lowercase the letters
        raw_data['text'] = raw_data['text'].apply(preprocess_text)
        # add commas to multilabels and change them  to numeric data
        raw_data['true_label'] = raw_data['true_label'].apply(preprocess_label)
        raw_data['pred_label'] = raw_data['pred_label'].astype(int)
        # add a tag for the source file
        raw_data['source'] = filename
        data_list.append(raw_data)
        print(time.time()-current_time)
        current_time = time.time()
    except:
        print("Error at ", filename, flush=True)

# get the most common label for each doc, and a tag that signifies if it is correct or not (type)
labelled_docs = all_labels(data_list)
#get only the rows with a label we're interested in
df_labelled = labelled_docs[labelled_docs.pred_label.isin(range(0,7))]
print("Docs labelled according to the most common label")

# read the keywords per class
df_HI = pd.read_csv(options.keyword_data+'/kw_stable_selec_filter_NEW.tsvHI.tsv', sep='\t')[['token']].rename(columns={"token": "HI"})
df_IN = pd.read_csv(options.keyword_data+'/kw_stable_selec_filter_NEW.tsvIN.tsv', sep='\t')[['token']].rename(columns={"token": "IN"})
df_ID = pd.read_csv(options.keyword_data+'/kw_stable_selec_filter_NEW.tsvID.tsv', sep='\t')[['token']].rename(columns={"token": "ID"})
df_IP = pd.read_csv(options.keyword_data+'/kw_stable_selec_filter_NEW.tsvIP.tsv', sep='\t')[['token']].rename(columns={"token": "IP"})
df_LY = pd.read_csv(options.keyword_data+'/kw_stable_selec_filter_NEW.tsvLY.tsv', sep='\t')[['token']].rename(columns={"token": "LY"})
df_NA = pd.read_csv(options.keyword_data+'/kw_stable_selec_filter_NEW.tsvNA.tsv', sep='\t')[['token']].rename(columns={"token": "NA"})
df_OP = pd.read_csv(options.keyword_data+'/kw_stable_selec_filter_NEW.tsvOP.tsv', sep='\t')[['token']].rename(columns={"token": "OP"})

# choose keyword_threshold of the best keywords
kw_list = [df_HI, df_ID, df_IN, df_IP, df_LY, df_NA, df_OP]
keywords = concatenate(kw_list, options.keyword_threshold)


# calculate the uniqueness of keywords between classes
#uniqueness(keywords=keywords)

# calculate the amount of keywords in documents, normalized by class
# (amount of keyword per doc) * (mean doc len with this label)/(current doc len)
coverage(df_labelled, keywords, options.coverage_threshold)

#corpus_coverage(keywords, df_labelled)















