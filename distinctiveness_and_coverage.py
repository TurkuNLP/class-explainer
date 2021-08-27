import numpy as np
import pandas as pd
import re
import glob
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import sys
import time
from collections import Counter
#from statistics import multimode
import csv

PATH = ''
NUMBER_OF_KEYWORDS = 100
key_values = ['HI', 'ID', 'IN', 'IP','LY', 'NA', 'OP']
PREDICTION_THRESHOLD = 0.5


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
    ap.add_argument('--prediction_threshold', type=float, default=PREDICTION_THRESHOLD,
                    help='Threshold for choosing best predictions from all predcitions')
    return ap



############################ PREPROCESSING ################################
def preprocess_text(d):
    # Separate punctuations from words by whitespace
    if len(d) == 0:
        d= " "
    d2 = re.sub(r"\s?([^\w\s'/\-\+$]+)\s?", r" \1 ", d)
    return d2.lower()
    


def preprocess_label(d):
    d2 = re.sub(r"\s?([^\w\s'/\-\+$]+)\s?", r" \1 ", d)
    new_label = [int(s) for s in d2.split() if s.isdigit()]
    new_label2 = [s for s in new_label if s < 7]
    return new_label2

def flatten(t):
    """
    Flattens a nested list to a normal list
    [[2,3], 4] = [2,3,4]
    """
    return [item for sublist in t for item in sublist]

def multimode(data):
    res = []
    data_list1 = Counter(data) 
    temp = data_list1.most_common(1)[0][1] 
    for ele in data:
        if data.count(ele) == temp:
            res.append(ele)
    res = list(set(res))

    return res 

def get_labels(df_list):
    """
    Calculates the best predicted label(s) for each doc.
    Also checks, if the prediction(s) is correct.
    Returns new dataframe with one row == one doc
    """


    data = []

    # concatenate all the dataframes
    df = pd.concat(df_list, axis = 0)

    # Loop over all the documents
    for doc in set(df['doc_id']):
        # get only the ones with this document id, and get the real label
        sub_df = df[df.doc_id == doc]
        true_labels = sub_df['true_label'].iloc[0]
        text = sub_df['text'].iloc[0]

        # get all possible predicted labels for this document
        pred_labels = []
        for index, row in sub_df.iterrows():
            pred_labels.append(row['pred_label'])

        pred_labels = flatten(pred_labels)
        
        # get the most common prediction(s) with multimode, and remove everything over 6
        #most_frequent_predictions_raw = np.array(multimode(pred_labels))
        #most_frequent_predictions = most_frequent_predictions_raw[most_frequent_predictions_raw < 7]

        n_experiments = len(df_list)
        cntr = Counter([x for x in pred_labels if x < 7])
        most_frequent_predictions = [cl for cl,cnt in cntr.items() if cnt >= n_experiments*options.prediction_threshold]



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
                if comp.isnull().iloc[index]:
                    values.append(None)
                else:
                    # if there is no match the word is unique
                    values.append(0)
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
    return sum(values == np.zeros(len(values)))/ len(values), sum(values == np.zeros(len(values))), len(values)



def distinctiveness(keywords):
    df = keywords
    percent = []
    percent_std = []
    percent_key = []

    print("Taking max ", len(keywords) ,"keywords: (fraction of distinct words, distinct words, all words)")
    # get frequencies
    value_matrix = get_value_matrix(df)
    percent_key = []
    # loop over them and print out info
    for i in range(len(value_matrix)):
        percent_key.append(get_0_percentage(value_matrix[i])[0])
        print(key_values[i], ": ", get_0_percentage(value_matrix[i])[0], "% , ", get_0_percentage(value_matrix[i])[1], "/", get_0_percentage(value_matrix[i])[2] )
    percent.append(np.mean(percent_key))
    print("mean: ",np.mean(percent_key))
    percent_std.append(np.std(percent_key))
    print("std dev: ",np.std(percent_key))
    return np.mean(percent_key)


  

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
        print(data['label'].value_counts())
        data.dropna(inplace = True)
        print(data['label'].value_counts())
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
    for word in keywords:
        try:
            if word in text.split(): #" "+word+" " in text:   # empty space to remove compound words ect.
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
    #print(labelled_predictions)
    df = process(labelled_predictions, style)

    
    # calculate text lengths (in unique words) for normalizing 
    df['text_length_in_char'] = df['text'].apply(lambda x: len(x))
    df['text_length'] = df['text'].apply(lambda x: len(np.unique(x.split(" "))))
    shortest_doc_len = min(df['text_length'])
    
    # mean text length, both options
    mean_text_length = get_mean_text_length(df)
    mean_text_length_in_char = get_mean_text_length_in_char(df)

    # theoretical max as 100 (max keywords) * highest max length wrt label / shortest doc overall
    theoretical_max_score = 100*max(mean_text_length)/shortest_doc_len


    # calculate the occurrences in docs, collect data
    l = []
    s = []
    t = []
    #c = []
    for index, row in df.iterrows():
        label_num = row['label']
        # loop over all labels
        for lb in label_num:
            label = key_values[lb]
            text = row['text']
            kw = keywords[label].values
            index = key_values.index(label)
    
            l.append(label)
            s.append(count_occurrence(kw, text)*(mean_text_length[index] / len(np.unique(text.split()))))
            t.append(row['text'])
            #c.append(row['type'])

    # if you want to save, uncomment these
    #save_df = pd.DataFrame(data = l, columns=['label'])
    #save_df['score'] = s
    #save_df['type'] = c
    #save_df.to_csv("coverage_scores2.tsv", sep="\t")
    

    # calculate mean per class
    coverage = pd.DataFrame(data=l, columns= ['label'])
    # divide score with theoretical max for normalisation
    coverage['score'] = np.array(s)/theoretical_max_score
    print(coverage['label'].value_counts())
    means = coverage.groupby('label').mean()
    print(means)



def corpus_coverage(keyword, labelled_predictions, style):
    data = process(labelled_predictions, style)
    
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
            scores.append(word_count/doc_count)
        #filename = "roskaa/Testttiiiuwu"+key+".csv"
        #with open(filename, "w") as f:
        #    for i in range(len(words)):
        #        line = str(words[i])+ ","+str(scores[i])+"\n"
        #        f.write(line)
        #f.close() 
        print(key, ": ", np.mean(scores), " support: ", doc_count)
        
        





if __name__=="__main__":

    options = argparser().parse_args(sys.argv[1:])
    data_list = []
    num_files = 0


    for filename in glob.glob(options.document_data+"/*.tsv"):
        try:
            num_files +=1
            print(filename, flush = True)
            raw_data = pd.read_csv(filename, sep='\t', names=['doc_id', 'pred_label', 'true_label', 'text'])#.rename(columns={"0":'doc_id', "1":'true_label', "2":'pred_label',"3":'text'}) # NOTE: Why is pred before true label?
            # remove null predictions
            print(raw_data.head())
            raw_data.dropna(axis = 0, how='any', inplace = True)
            # add white space to punctuation and lowercase the letters
            raw_data['text'] = raw_data['text'].apply(preprocess_text)
            # add commas to multilabels and change them  to numeric data
            raw_data['true_label'] = raw_data['true_label'].apply(preprocess_label)
            raw_data['pred_label'] = raw_data['pred_label'].apply(preprocess_label)
            # add a tag for the source file
            raw_data['source'] = filename
            data_list.append(raw_data)
        except:
            print("Error at ", filename, flush=True)


    # get the most common prediction for all documents
    # prediction must be 0...6
    # if there are labels between 0 and 6 that are equally good
    # keep them both
    labelled_docs = get_labels(data_list)
    #print(labelled_docs.head(30))


    # read the keywords per class
    df_HI = pd.read_csv(options.keyword_data+'.tsvHI.tsv', sep='\t')[['token']].rename(columns={"token": "HI"})
    df_IN = pd.read_csv(options.keyword_data+'.tsvIN.tsv', sep='\t')[['token']].rename(columns={"token": "IN"})
    df_ID = pd.read_csv(options.keyword_data+'.tsvID.tsv', sep='\t')[['token']].rename(columns={"token": "ID"})
    df_IP = pd.read_csv(options.keyword_data+'.tsvIP.tsv', sep='\t')[['token']].rename(columns={"token": "IP"})
    df_LY = pd.read_csv(options.keyword_data+'.tsvLY.tsv', sep='\t')[['token']].rename(columns={"token": "LY"})
    df_NA = pd.read_csv(options.keyword_data+'.tsvNA.tsv', sep='\t')[['token']].rename(columns={"token": "NA"})
    df_OP = pd.read_csv(options.keyword_data+'.tsvOP.tsv', sep='\t')[['token']].rename(columns={"token": "OP"})

    kw_list = [df_HI, df_ID, df_IN, df_IP, df_LY, df_NA, df_OP]
    keywords = concatenate(kw_list, options.number_of_keywords)

    # NOW all is preprocessed
    print("Preprocessing done", flush=True)





    # THE CALCULATIONS

    distinctive_mean = distinctiveness(keywords=keywords)

    coverage(labelled_docs, keywords, options.style)

    corpus_coverage(keywords, labelled_docs, options.style)



