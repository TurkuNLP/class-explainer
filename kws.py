import numpy as np
import glob
import time
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import sys
from numpy.core.numeric import NaN
import pandas as pd
import warnings
import json

if not sys.warnoptions:
    warnings.simplefilter("ignore")



# HYPERPARAMETRES
CHOOSE_BEST = 20
DROP_AMBIGUOUS = 3
FRACTION = 0.6
STD_THRESHOLD = 0.2
THRESHOLD = 3
SAVE_N = 50
QUANTILE = 0.25
SAVE_FILE = "testi2_stable_keywords.tsv"
UNSTABLE_FILE = "testi2_unstable_keywords.tsv"


def argparser():
    ap = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    ap.add_argument('--data', metavar='FILE', required=True,
                    help='Path to data. /* already included in call')
    ap.add_argument('--choose_best', metavar='INT', type=int,
                    default=CHOOSE_BEST, help='Number of best words chosen per each doc-label pair')
    ap.add_argument('--filter', required=True, metavar='FILE',
                    help='method for filtering, std or selectf')
    ap.add_argument('--fraction', metavar='FLOAT', type=float,
                    default=FRACTION, help='% in how many lists the word must be present in (selection frequency')
    ap.add_argument('--std_threshold', metavar='FLOAT', type=float, 
                    default=STD_THRESHOLD, help='Threshold for std filtering')
    ap.add_argument('--threshold', metavar='INT', type=int,
                    default=THRESHOLD, help='Threshold for dropping words that are too rare')
    ap.add_argument('--save_n', metavar='INT', type=int,
                    default=SAVE_N, help='How many words per class are saved')
    ap.add_argument('--save_file', default=SAVE_FILE, metavar='FILE',
                    help='dir+file for saving the results')
    ap.add_argument('--unstable_file', default=UNSTABLE_FILE, metavar='FILE',
                    help='dir+file for saving the unstable results')
    return ap

def process(data):
    """
    Remove errors in data.
    """
    data['token'] = data['token'].str.replace('[^\w\s]','')
    data.replace("", NaN, inplace=True)
    data.dropna(axis=0, how='any', inplace=True)
    #print(data[data['score']>1])
    data.drop(data.index[data['score']> 1], inplace =True)


def read_data(data_name):
    """ read the data from a csv and remove null values (no predictions)"""
    
    data = pd.read_csv(data_name, delimiter = "\t", quotechar="¤", names = ['document_id', 'pred_label', 'token', 'score'])
    data['token'] = data['token'].str.lower()
    #data['token'] = data['token'].fillna("NaN_")
    data.dropna(axis=0, how='any', inplace=True)
    data = data[data.pred_label != "None"]
    process(data)

    return data


def choose_n_best(data, n):
    """ choose n best scoring words per document """
    df_new = data.sort_values('score', ascending=False).groupby(['document_id', 'pred_label']).head(n)
    df_new.sort_index(inplace=True)
    return df_new


def class_frequencies(data):
    """
    Calculate mean, std of scores and number of sources
    """
    #a = data.groupby(['token'])['pred_label'].unique()
    a = data.groupby(['token','pred_label'])['score'].unique()
    b = data.groupby(['token','pred_label'])['score'].mean()
    b2 = data.groupby(['token','pred_label'])['score'].std(ddof=0)
    c = data.groupby(['token','pred_label'])['source'].unique()
    
    return pd.concat([a.to_frame(), b.to_frame().add_suffix("_mean"), b2.to_frame().add_suffix("_std"), c.to_frame()],axis=1)
           
  
def flatten(t):
    """
    Flattens a nested list to a normal list
    [[2,3], 4] = [2,3,4]
    """
    return [item for sublist in t for item in sublist]


def filter_class_df(data, key, number):
    """
    Take out everything that has a low document frequency wrt current label (key, number).
    This also removes all tokenization errors, since their document frequency
    should always be 0.
    Save to a file.
    """
    with open('class_df.json') as f:   #../../samuel/class-explainer/
        dataset = json.load(f)
    # class_df contains data like this:
    # {HI: {baking: 5, recipe: 5, apple: 2},
    #  IN: {abstract: 4, information: 3, contains: 4},
    # ...}

    # inver the mapping. Original is not an injection, thus we get lists as outputs
    # eg. for HI:
    # 5: [baking, recipe], 2: [apple]
    inv_map = {}
    for k, v in dataset[key].items():
        inv_map[v] = inv_map.get(v, []) + [k] 
  
    # take words that have frequency over a specified threshold
    # and flatten the list for easy iteration
    common_words = flatten([inv_map[x] for x, y in inv_map.items() if x > options.threshold])

    # take a subset of data wrt the current label
    data2 = data[data.pred_label == number]
    # take out all that is in common words
    data_save = data2[data2['token'].isin(common_words)]
    # save also the complement
    data_errors = data2[~data2['token'].isin(common_words)]


    # save to files (pred_label not needed since one file contains only one label)
    filename = options.save_file+key+".tsv"
    filename_err = options.save_file+key+"_err.tsv"
    data_save.drop(['pred_label'], axis=1, inplace=True)
    data_errors.drop(['pred_label'], axis=1, inplace=True)

    data_save.to_csv(filename, sep="\t")  
    data_errors.to_csv(filename_err, sep="\t")  




if __name__=="__main__":
    #print("kws.py",flush = True)
    options = argparser().parse_args(sys.argv[1:])
    print(options,flush = True)

    # to see progression speed
    start = time.time()
    current = time.time()

    # data collection:
    # num_files keeps count on the number of files that
    # are succesfully opened
    df_list = []
    num_files = 0

    for filename in glob.glob(options.data+"*s.tsv"):
        try:
            num_files +=1
            print(filename, flush = True)
            # read data and correct null values
            df = read_data(filename)
            # choose a subset of best values per document-prediction pair
            df = choose_n_best(df, options.choose_best)  
            # append a source tag to each separate file
            df['source'] = filename
            print(time.time()-current, flush=True)
            current = time.time()
            # append data to list
            df_list.append(df)
        except:
            print("Error at ", filename, flush=True)
            current = time.time()
        
    print("Preprocessing time: ", time.time()-start, flush=True)
    
    
    # concatenate all for further analysis
    df_full = pd.concat(df_list)
    
    # get statistic of scores for each word-prediction pair
    # for example:
    # token       label   score        source
    # mouse       5       [0.5]        [file2]
    # mouse       5       [0.4]        [file4]
    # mouse       3       [0.2]        [file2]
    # turns into
    # token       label   score            score_mean   score_std     source
    # mouse       5       [0.5, 0.4]       [0.45]       [0.05]        [file2, file4]
    # mouse       3       [0.2]            [0.2]        [0.00]        [file2]
    freq_array = class_frequencies(df_full)
    
    # filter the data according to a method specified in the parametres
    # we also save everything that is filtered out in df_unstable
    print("Filtering", flush=True)
    if options.filter == 'std':
        # we filter out everything where the score_std is higher than the threshold
        df_save = freq_array[freq_array.score_std < options.std_threshold]
        df_unstable = freq_array[freq_array.score_std >= options.std_threshold]
    if options.filter == 'selectf':
        # here we look at the selection frequency e.g. how many separate sources the word is in
        # and drop if it is in less than the specified threshold (fraction)
        df_save = freq_array[freq_array['source'].apply(lambda x: len(x) >= options.fraction*num_files)]
        df_unstable = freq_array[freq_array['source'].apply(lambda x: len(x) < options.fraction*num_files)] 

    # sort the values by label and mean score, and take a certain amount of best results per label
    print("Sorting", flush=True)
    new_df_save = df_save.sort_values(['pred_label','score_mean'], ascending=[True, False]).groupby(['pred_label'],as_index=False).head(options.save_n)
    
    # we add a column that contains the number of sources
    # and drop the raw score and source columns
    new_df_save['source_number'] = new_df_save['source'].apply(lambda x: len(x))
    new_df_save.drop(['score','source'], axis=1, inplace=True)
    df_unstable.drop(['score','source'], axis=1, inplace=True)

    # removing pandas hierarchy for extractiong labels
    # groupby() makes the (token, pred_label) pair be an index of the column, so it cant be easily referenced
    # take the index values (token, pred_label) and make them into a new frame
    df_sep = pd.DataFrame(data= new_df_save.index.values.tolist(), columns = ['token', 'pred_label'])
    # add all other data
    df_sep['score_mean'] = new_df_save['score_mean'].to_numpy()
    df_sep['score_std'] = new_df_save['score_std'].to_numpy()
    df_sep['source_number'] = new_df_save['source_number'].to_numpy()

    # same for the other frame
    df_sep_unstable = pd.DataFrame(data= df_unstable.index.values.tolist(), columns = ['token', 'pred_label'])
    df_sep_unstable['score_mean'] = df_unstable['score_mean'].to_numpy()
    df_sep_unstable['score_std'] = df_unstable['score_std'].to_numpy()

    # save unstable words
    #for key, number in {'HI': 0, 'ID':1, 'IN':2,'IP':3,'LY':4,'NA':5,'OP':6,'SP':7}.items():
    #    filename = options.unstable_file+key+".tsv"
    #    df_sep_unstable['token'].to_csv(filename, sep="\t")  

    # save results for all the labels separately
    print("Saving individuals with rare words removed", flush=True)
    filter_class_df(df_sep,"HI", 0)
    filter_class_df(df_sep,"ID", 1)
    filter_class_df(df_sep,"IN", 2)
    filter_class_df(df_sep,"IP", 3)
    filter_class_df(df_sep,"LY", 4)
    filter_class_df(df_sep,"NA", 5)
    filter_class_df(df_sep,"OP", 6)
    filter_class_df(df_sep,"SP", 7)

    print("Everything done", flush=True)
    print(time.time()-start, flush=True)
