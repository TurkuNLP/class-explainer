import pandas as pd
import numpy as np
from scipy.stats import rankdata
import glob
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import sys
import itertools


# HYPERPARAMETRES
CHOOSE_BEST = 10
DROP_AMBIGUOUS = 3
FRACTION = 0.8
SAVE_N = 100
QUANTILE = 0.25
SAVE_FILE = "keywords.tsv"


def argparser():
    ap = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    ap.add_argument('--data', metavar='FILE', required=True,
                    help='Path to data. /* already included in call')
    ap.add_argument('--language', metavar='FILE', required=True,
                    help='Language, together with data: data /* language.tsv')
    ap.add_argument('--choose_best', metavar='INT', type=int,
                    default=CHOOSE_BEST, help='Number of best words chosen per doc')
    ap.add_argument('--drop_amb', metavar='INT', type=int, default=DROP_AMBIGUOUS,
                    help='Upper limit for classes a word can be present in')
    ap.add_argument('--fraction', metavar='FLOAT', type=float,
                    default=FRACTION, help='% in how many lists the word must be present in')
    ap.add_argument('--quantile',metavar='FLOAT', type=float,
                    default=QUANTILE, help='Quantile for dropping words')
    ap.add_argument('--save_n', metavar='INT', type=int,
                    default=SAVE_N, help='How many words/class are saved')
    ap.add_argument('--save_file', default=SAVE_FILE, metavar='FILE',
                    help='dir+file for saving the results')
    return ap


def read_data(data_name):
    """ read the data from a csv and remove null values (no predictions)"""

    data = pd.read_csv(data_name, delimiter = "\t", names = ['id','document_id', 'real_label', 'pred_label', 'token', 'score', 'logits'],index_col=False)
    data['token'] = data['token'].str.lower()
    #data['token'] = data['token'].fillna("NaN_")
    data.dropna(axis=0, how='any', inplace=True)
    data = data[data.pred_label != "None"]

    return data

def remove_false_predictions(data):
    """ remove all false predictions """
    drop_column = []
    for index, row in data.iterrows():
        if (row['pred_label'] in row['real_label']):
            drop_column.append(True)
        else:
            drop_column.append(False)
    data['drop_column'] = drop_column
    new_data = data[data.drop_column == True]
    new_data.drop(['drop_column'], axis=1, inplace=True)
    return new_data
    
            

def choose_n_best(data, n):
    """ choose n best scoring words per document """
    df_new = data.sort_values('score', ascending=False).groupby(['document_id', 'pred_label']).head(n)
    df_new.sort_index(inplace=True)
    return df_new

#def choose_n_best(data, n):
#    """ choose n best scoring words per document """
#
#    df_topscores = pd.DataFrame(columns = ['document_id', 'real_label','pred_label','token', 'score'])#, 'logits']) 
#    for doc in (set(data['document_id'])):
#        df = data[data.document_id == doc]
#        df_topscores = pd.concat([df_topscores,df.nlargest(n, 'score')], axis = 0)
#
#    df_topscores.sort_index(inplace=True)
#
#    return df_topscores


def get_frequencies(df_topscores):
    """ calculate the frequencies of each word-label combination and append them to the dataframe under freq """

    freq_df = df_topscores.groupby(['token']).count()

    frequencies = []
    index = 0
    for token in df_topscores['token']:
        try:
            #print(freq_df['document_id'][str(token)])
            frequencies.append(freq_df['document_id'][str(token)])
        except:
            print("Error with freq calculations, at index ", index, ", token: ", token)
            break
        index += 1
 
    df_topscores['freq'] = frequencies

def get_sourcefrequencies(df_topscores):
    source_numbers = []
    
    for index, row in df_topscores.iterrows():
        source_list = list(df_topscores[df_topscores.token == row['token']]['source'])
        source_numbers.append(len(set(source_list)))

    df_topscores['source_freq'] = source_numbers

def get_classfrequencies(df_topscores):
    label_numbers = []
    label_set = []
    for index, row in df_topscores.iterrows():
        classes_list = list(df_topscores[df_topscores.token == row['token']]['pred_label'])
        label_numbers.append(len(set(classes_list)))
        label_set.append(set(classes_list))

    df_topscores['class_freq'] = label_numbers
    df_topscores['class_set'] = label_set


def drop_ambiguous_words(df, n):
    """ drop words that are present in too many classes """

    get_classfrequencies(df)
    df_dropped = df[df.class_freq < n]
    return df_dropped

def lin(x):
    """ calculate linear scores to words """

    maxim = max(x)  # x_1 3 -> y_1 1
    minim = min(x)   # x_0 1  -> y_0 0
    k = 1.0/(maxim-minim)
    # y-y_0 = k*(x-x_0)
    # y = k*(x-x0) +y_0 
    return np.array(k*(x-minim)+0)
  


def rank_f(df, doc_id):
    """ function for ranking the data """

    scores = np.array(df)
    ranked = rankdata(scores, method='dense')
    return lin(ranked)


def rank(df_topscores):
    """ assign ranks to words and add them to the dataframe under rank """
    ranks = []

    for doc_id in set(df_topscores['document_id']): 
        #print(doc_id)

        df = df_topscores[df_topscores.document_id == doc_id]['score']
        if len(df) == 0:
          print(doc_id)
        r = rank_f(df,doc_id)
        for item in r:
            ranks.append(float(item))
    
    df_topscores['rank'] = np.array(ranks)

if __name__=="__main__":
    options = argparser().parse_args(sys.argv[1:])
    # get all data in a list
    df_list = []

    num_files = 0
    for filename in glob.glob(options.data+"/*"+options.language+".tsv"):
        num_files += 1
        print(filename)
        df = read_data(filename)
        df.drop(['id'], axis=1, inplace=True)
        #df = df[df.pred_label in df.real_label]
        df = remove_false_predictions(df)
        df = choose_n_best(df, options.choose_best)
        
        get_classfrequencies(df)
        rank(df)
        df['source'] = filename
        df_list.append(df)
    
    df_full = pd.concat(df_list)


    # 1) we want to identify which words are stable across many runs, 
    # and which happen to get a high ranking due to non-representative split, and 
    # 2) we might want to keep some sense of relative importance within the list, 
    # after filtering out spurious words


    ##### STABLE ACROSS RUNS ######
    ##### -> Present almost always (fraction %)  #####

    #get_sourcefrequencies(df_full)
    #print(df_full)
    
    #all_kws = []
    #all_kws.append(np.array(df_full['token']))
    #all_kws = np.unique(np.array(all_kws)).flatten()

    all_lbs = []
    all_lbs.append(np.array(df_full['pred_label']))
    all_lbs = np.unique(np.array(all_lbs)).flatten()

    kw = []
    for label in all_lbs:
        df_l = df_full[df_full.pred_label == label]
        all_kws = set(df_l['token'])
        kw.append(all_kws)
    
    save_list = []
    for label, wordlist in zip(all_lbs, kw):
        for word in wordlist:
            df_sub = df_full[(df_full.token == word)&(df_full.pred_label == label)]
            # check if word+prediction in sufficiently many model predictions:
            if len(set(df_sub['source'])) >= options.fraction*num_files:
                df_sub['freq'] = len(set(df_sub['source']))
                a = df_sub['rank'].quantile(options.quantile)
                b = df_sub['rank'].quantile(1-options.quantile)
                #if len(save_list) < 5:
                #    print("Quantiles: ", a, b)
                #    print(df_sub)
                df_sub = df_sub.drop(df_sub.index[(df_sub['rank'] < a)])
                df_sub = df_sub.drop(df_sub.index[(df_sub['rank'] > b)])
                #if len(save_list) < 5:
                #    print(df_sub)
                save_list.append(df_sub)
    
    
    df_comp = pd.concat(save_list)
    df_comp.sort_values(['pred_label', 'rank'], ascending=[True, False], inplace=True)
    df_save = df_comp.groupby('pred_label').head(options.save_n)
    df_save.drop(['logits','source','class_set'], axis = 1, inplace=True)
   
    #print(df_save[df_save.pred_label == "0"])
    #print(df_save[df_save.pred_label == "1"])
    #print(df_save[df_save.pred_label == "2"])
    #print(df_save[df_save.pred_label == "3"])


    df_save.to_csv(options.save_file, sep="\t")
    
