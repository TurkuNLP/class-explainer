import pandas as pd
import numpy as np
from scipy.stats import rankdata
import glob
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import sys


# HYPERPARAMETRES
CHOOSE_BEST = 10
DROP_AMBIGUOUS = 3
FRACTION = 0.8
SAVE_N = 100
SAVE_FILE = "keywords.tsv"


def argparser():
    ap = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    ap.add_argument('--data', metavar='FILE', required=True,
                    help='Path to data. /*.tsv already included in call')
    ap.add_argument('--choose_best', metavar='INT', type=int,
                    default=CHOOSE_BEST, help='Number of best words chosen per doc')
    ap.add_argument('--drop_amb', metavar='INT', type=int, default=DROP_AMBIGUOUS,
                    help='Upper limit for classes a word can be present in')
    ap.add_argument('--fraction', metavar='FLOAT', type=float,
                    default=FRACTION, help='% in how many lists the word must be present in')
    ap.add_argument('--save_n', metavar='INT', type=int,
                    default=SAVE_N, help='How many words/class are saved')
    ap.add_argument('--save_file', default=SAVE_FILE, metavar='FILE',
                    help='dir+file for saving the results')
    return ap




def read_data(data_name):
    """ read the data from a csv and remove null values (no predictions)"""

    data = pd.read_csv(data_name, delimiter = "\t", names = ['id','document_id', 'real_label', 'pred_label', 'token', 'score', 'logits'], index_col=False)
    data['token'] = data['token'].str.lower()
    #data['token'] = data['token'].fillna("NaN_")
    data.dropna(axis = 0, inplace=True)

    return data

def choose_n_best(data, n):
    """ choose n best scoring words per document """
    df_new = data.sort_values('score', ascending=False).groupby('document_id').head(n)
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
    """ calculate the frequencies of each word and append them to the dataframe under freq """

    freq_df = df_topscores.groupby('token').count()

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

    for doc_id in range(len(set(df_topscores['document_id']))): 
        #print(doc_id)
        df = df_topscores[df_topscores.document_id == 'document_'+str(doc_id)]['score']
        
        r = rank_f(df,doc_id)
        for item in r:
            ranks.append(item)
    
    df_topscores['rank'] = np.array(ranks)

    
    
    
    
if __name__=="__main__":
    options = argparser().parse_args(sys.argv[1:])

    # get all data in a list
    df_list = []

    for filename in glob.glob(options.data+"/*.tsv"):
        print(filename)
        df = choose_n_best(read_data(filename),options.choose_best)
        get_frequencies(df)
        df = drop_ambiguous_words(df, options.drop_amb)
        rank(df)
        df_list.append(df)

    # all keywords present in any list
    all_kws = []
    for i in range(len(df_list)):
        all_kws.append(np.array(df_list[i]['token']))
    all_kws = np.unique(np.array(all_kws).flatten()) 

    # one concatenated list for calculating statistics
    df_full = pd.concat(df_list, ignore_index=True)

    # array for the good keywords
    keywords = []

    for word in all_kws:
        # counter to see in how many lists the words is in
        counter = 0
        # if word in list, counter ++
        for i in range(len(df_list)):
            if word in np.array(df_list[i].token):
                counter += 1
        # if word was present almost always
        if counter >= options.fraction*len(df_list):
            # select those words from full dataframe
            df_sub = df_full[(df_full.token == word)]
            # for all labels predicted for that word (max given before)
            for label in set(df_sub['pred_label']):
                # get another sub dataframe that has only that label
                df_sub2 = df_sub[df_sub.pred_label == label]
                # if there are predictions, calculate statistics
                if len(df_sub2) >0:
                  fre = len(df_sub2)
                  classes = set(df_sub['pred_label'])
                  mean = df_sub2['rank'].mean()
                  std = df_sub2['rank'].std()
                  min = df_sub2['rank'].min()
                  max = df_sub2['rank'].max()
                  keywords.append([label, word, fre, classes, mean, std, min, max]) 

    # make a dataframe, sort the keywords wrt label and mean rank
    df_comp = pd.DataFrame(data=keywords, columns = ['label','word', 'freq', 'class_freq', 'mean', 'std', 'min', 'max'])      
    df_comp.sort_values(['label', 'mean'], ascending=[True, False], inplace=True)
    df_save = df_comp.groupby('label').head(options.save_n)


    df_save.to_csv(options.save_file, sep="\t")
