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
    ap.add_argument('--language', metavar='FILE', default="",
                    help='Language, together with data: data /* language.tsv')
    ap.add_argument('--choose_best', metavar='INT', type=int,
                    default=CHOOSE_BEST, help='Number of best words chosen per doc')
    ap.add_argument('--drop_amb', metavar='INT', type=int, default=DROP_AMBIGUOUS,
                    help='Upper limit for classes a word can be present in')
    ap.add_argument('--fraction', metavar='FLOAT', type=float,
                    default=FRACTION, help='% in how many lists the word must be present in')
    ap.add_argument('--quantile',metavar='FLOAT', type=float,
                    default=QUANTILE, help='Quantile for dropping words')
    ap.add_argument('--drop_false_predictions',metavar='INT', type=int,
                    default=1, help='Whether or not to drop false predictions, 0/1')
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

if __name__=="__main__":
    print("stable_keywords.py",flush = True)
    options = argparser().parse_args(sys.argv[1:])
    print(options, flush = True)
    # get all data in a list
    df_list = []

    num_files = 0
    for filename in glob.glob(options.data+"*.tsv"):
        num_files += 1
        print(filename, flush = True)
        df = read_data(filename)
        df.drop(['id'], axis=1, inplace=True)
        df['score'] = pd.to_numeric(df['score'])
        #df = df[df.pred_label in df.real_label]
        if options.drop_false_predictions==1:
            df = remove_false_predictions(df)
            print("False predictions removed",flush = True)
        df = choose_n_best(df, options.choose_best)
        get_classfrequencies(df)
        df['source'] = filename
        df_list.append(df)

    df_full = pd.concat(df_list)


    all_lbs = []
    all_lbs.append(np.array(df_full['pred_label']))
    all_lbs = np.unique(np.array(all_lbs)).flatten()

    kw = []
    for label in all_lbs:
        df_l = df_full[df_full.pred_label == label]
        all_kws = set(df_l['token'])
        kw.append(all_kws)
    print("Looping over keywords", flush = True)
    save_list = []
    for label, wordlist in zip(all_lbs, kw):
        for word in wordlist:
            df_sub = df_full[(df_full.token == word)&(df_full.pred_label == label)]
            # check if word+prediction in sufficiently many model predictions:
            if len(set(df_sub['source'])) >= options.fraction*num_files:
                df_sub['freq'] = len(set(df_sub['source']))
                save_list.append(df_sub)
    
    df_save = pd.concat(save_list)
    df_save.sort_values(['pred_label', 'freq', 'score'], ascending=[True, False, False], inplace=True)
    df_save.drop_duplicates(subset=['token', 'pred_label'], keep="first", inplace=True)
    df_save.drop(['logits','source'], axis=1, inplace=True)
    
    #print(df_save)
    df_save.to_csv(options.save_file, sep="\t")
    print("Saved succesfully", flush = True)
