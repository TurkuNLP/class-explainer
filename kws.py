import numpy as np
import csv
import glob
import time
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import sys
from numpy.core.defchararray import title
from numpy.core.numeric import NaN
import pandas as pd
import warnings
import json
import matplotlib.pyplot as plt

if not sys.warnoptions:
    warnings.simplefilter("ignore")

csv.field_size_limit(sys.maxsize)


# HYPERPARAMETRES
WORDS_PER_DOC = 20
DROP_AMBIGUOUS = 3
SELECTION_FREQ = 0.6
STD_THRESHOLD = 0.2
MIN_WORD_FREQ = 3
SAVE_N = 1000
QUANTILE = 0.25
SAVE_FILE = "testi2_stable_keywords.tsv"
UNSTABLE_FILE = "testi2_unstable_keywords.tsv"


def argparser():
    ap = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    ap.add_argument('--data', metavar='FILE', required=True,
                    help='Path to data. /* already included in call')
    ap.add_argument('--words_per_doc', metavar='INT', type=int, default=WORDS_PER_DOC,
                    help='Number of best words chosen per each doc-label pair. Optimize')
    ap.add_argument('--filter', metavar='FILE', default = 'selectf',
                    help='method for filtering, std or selectf')
    ap.add_argument('--selection_freq', metavar='FLOAT', type=float,
                    default=SELECTION_FREQ, help='% in how many lists the word must be present in (selection frequency). Optimize')
    ap.add_argument('--min_word_freq', metavar='INT', type=int, default=MIN_WORD_FREQ,
                    help='Threshold for dropping words that are too rare')
    ap.add_argument('--save_n', metavar='INT', type=int, default=SAVE_N,
                    help='How many words per class are saved. This needs to be really high, explanation in comments')
    ap.add_argument('--save_file', default=SAVE_FILE, metavar='FILE',
                    help='dir+file for saving the the keywords')
    ap.add_argument('--unstable_file', default=UNSTABLE_FILE, metavar='FILE',
                    help='dir+file for saving the unstable results')
    ap.add_argument('--keyword_data', metavar='FILE', required=True,
                    help='Path to keyword data. SAME AS save_file')
    ap.add_argument('--document_data', metavar='FILE', required=True,
                    help='Path to document data, ALL DOCS TEXTS AND PREDICTIONS')
    ap.add_argument('--number_of_keywords', metavar='INT', type=int, default=NUMBER_OF_KEYWORDS,
                    help='Threshold for number of keywords compared/chosen per register. FIXED 100')
    ap.add_argument('--style', metavar='STR', type=str, default='TP',
                    help='TP = True Positive, P = Predictions, TL = True Label')
    ap.add_argument('--prediction_th', type=float, default=PREDICTION_THRESHOLD,
                    help='Threshold on model posterior probability. For logging purposes only, threshold is controlled by --data.')
    ap.add_argument('--frequent_predictions_th', type=float, default=FREQUENT_PREDICTIONS_THRESHOLD,
                    help='Threshold for choosing best predictions from all predcitions. Maybe optimizable?')

    ap.add_argument('--std_threshold', metavar='FLOAT', type=float,
                    default=STD_THRESHOLD, help='Threshold for std filtering')
    ap.add_argument('--plot_file', metavar='FILE', required=True,
                    help='File for saving plots')
    return ap

def process(data):
    """
    Remove errors in data.
    """
#    print("l 71", data.head(3))
    data['token'] = data['token'].str.replace('[^\w\s]','')
    data.replace("", NaN, inplace=True)
#    print("l 74", data.head(3))
    data.dropna(axis=0, how='any', inplace=True)
#    data.drop(data.index[data['score']> 1], inplace =True) # this just breaks
 #   print("l80 ind", data.head(3))

def read_data(data_name):
    """ read the data from a csv and remove null values (no predictions)"""
  #  print("YYY read data firing")
  #  print("YYY data_name", data_name)
    data = pd.read_csv(data_name, delimiter = "\t", quotechar="¤", names = ['document_id', 'pred_label', 'token', 'score'], error_bad_lines=False)
 #   print("YYY data new", data.head(5))
    data['token'] = data['token'].str.lower()
    #data['token'] = data['token'].fillna("NaN_")
    data.dropna(axis=0, how='any', inplace=True)
    data = data[data.pred_label != "None"]
    process(data)
  #  print("filename", filename, "read l 87")
   # print("YYY data l 88", data.head(5))
    return data

def read_data_v(data_name):
    full = []
    for line in open(data_name, "r", encoding="utf-8"):
        new = []
        line=line.strip().split("\t")
        new.append(line[0])
        new.append(line[1])
        new.append(line[2])
        new.append(float(line[3]))
        if float(line[3]) > 1.0:
            print("ZZZZ line", line)
        full.append(new)
    data=pd.DataFrame(full,  columns = ['document_id', 'pred_label', 'token', 'score'])
#    print("data index l 103", data.index)
    data['token'] = data['token'].str.lower()
    #data['token'] = data['token'].fillna("NaN_")
    data.dropna(axis=0, how='any', inplace=True)
    data = data[data.pred_label != "None"]
    process(data)
 #   print("data idex l 109", data.index)
  #  print(data.head(1))
    return data    
        
def choose_n_best(data, n):
    """ choose n best scoring words per document """
    #print("N", n)
    df_new = data.sort_values('score', ascending=False).groupby(['document_id', 'pred_label']).head(n)
   # df_new.to_csv("df_new.csv")
 #   print("choose new", df_new.index)
  #  print("ranked", df_new.head(5))
    df_new.sort_index(inplace=True)
   # df_new.to_csv("new_sorted.csv")
#    print("2", df_new.index)
#    print("3", df_new.head(5))
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
q    [[2,3], 4] = [2,3,4]
    """
    return [item for sublist in t for item in sublist]


def filter_class_df(data, key, number, options):
    """
    Take out everything that has a low document frequency wrt current label (key, number).
    This also removes all tokenization errors, since their document frequency
    should always be 0.
    Save to a file.
    """
    #with open('class_df.json') as f:   #../../samuel/class-explainer/
    with open(options.class_df) as f:   #../../samuel/class-explainer/
        dataset = json.load(f)
#        print("XXX dataset", dataset)
    # class_df contains data like this:
    # {HI: {baking: 5, recipe: 5, apple: 2},
    #  IN: {abstract: 4, information: 3, contains: 4},
    # ...}

    # inver the mapping. Original is not an injection, thus we get lists as outputs
    # eg. for HI:
    # 5: [baking, recipe], 2: [apple]
    
    #u'\xe4'.encode('ascii','ignore')
    inv_map = {}
    delme=[]
    print("TRYIONG data l 143 KEY NUMBER", key, number)
#    print("DATASET", dataset, flush=True)
#    try:
#    print("HEAD", dataset[key].head(5))
#    print("NO")
#    [print(v) for i, v in enumerate(dataset.items()) if i < 5]
    for k, v in dataset[key].items():
#            print("KV", k)
 #           print("KVV", v)
 #           print("key", key, flush=True)
      #  print("XXX kv", k.encode('utf-8','ignore'))
       # print("XXX v", v)
    
            inv_map[v] = inv_map.get(v, []) + [k]
            
    # take words that have frequency over a specified threshold
    # and flatten the list for easy iteration
 #   print("DICT HEAD", inv_map.head(10))
    common_words = flatten([inv_map[x] for x, y in inv_map.items() if x > options.min_word_freq])
    fout = open(key+"_common_words.txt", "w")
    for x in common_words:
        fout.write(x)
    fout.close()
    # take a subset of data wrt the current label
#    data2 = data[data.pred_label == number]
    data.to_csv("beforedata2.csv")
   # data2 = data.loc[data["pred_label"] == int(number)]
    data2_v = pd.DataFrame()
   # print("NUM", number)
    for i in range(len(data)):
       # print("1", data.iloc[i,0], data.iloc[i,1], data.iloc[i,2])
    #    print("2", data.iloc[i,1])
        if int(data.iloc[i,1]) == int(number):
     #       print("jee")
            data2_v = data2_v.append(data.iloc[i])
    print("index new", data2_v.index)
#    for row in data.iterrows():
#        print("1", row)
#        print("type", type(row))
#        print("2", row['pred_label'])
#        if int(row[2]) == int(number):
#            print("founD")
    print("head 5", data.head(5))
    print("data 2", data2_v.head(5))
    # take out all that is in common words
    data_save = data2_v[data2_v['token'].isin(common_words)]
    # save also the complement
    print("datasave len", data_save.index)
    data_errors = data2_v[~data2_v['token'].isin(common_words)]
    print("erors", data_errors.index)

    # save to files (pred_label not needed since one file contains only one label)
    filename = options.save_file+key+".tsv"
#    print("filename l 164 kw", filename)
    filename_err = options.save_file+key+"_err.tsv"
    data_save.drop(['pred_label'], axis=1, inplace=True)
    data_errors.drop(['pred_label'], axis=1, inplace=True)

    data_save.to_csv(filename, sep="\t")
    data_errors.to_csv(filename_err, sep="\t")
   # except:
   #     if KeyError:
    #        print("keyvirhe")
     #   else:
      #      print("joku muu")
#        if KeyError:
 #           continue





def process_data(options):
    df_list = []
    num_files = 0
    
    for filename in glob.glob(options.data+"*s.tsv"):
        #try:
        num_files +=1
 #       print(filename, "l 189 kws.py", flush = True)
        # read data and correct null values
        try:
            df = read_data_v(filename)
            print("XXX filename success 194", filename, flush=True)
        except:
            print("l 197 read data ERROR", filename, flush=True)
     #   print("l 221", df.index, flush=True)
#        try:
#        print("trying l 223", flush=True)
#            print("XXX filename", filename)
        # choose a subset of best values per document-prediction pair
        df = choose_n_best(df, options.words_per_doc)
      #  print("df after best", df.index, flush=True)
        print(" after best head", df.head(3))
        # append a source tag to each separate file
        df['source'] = filename
        print("source?", df.head(5))
        #print(time.time()-current, flush=True)
        #current = time.time()
        # append data to list
        df_list.append(df)
#        print("l 203, dflist", df_list[0], flush=True)
 #       print("l 203, dflist", df_list[-1], flush=True)
        #except:
        #    print("Error at ", filename, flush=True)
        #    current = time.time()
 #       except:
        #    print("XXX filename kws.py l 207", filename, flush=True)


    # concatenate all for further analysis
#    print("len df list l 240", len(df_list))
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
    df_full = df_full[df_full['token'].apply(lambda x: any([y.isalpha() for y in x]))] # Filter out tokens without any letter
    freq_array = class_frequencies(df_full)
 #   print("df full", df_full.head(5))
  #  print("freq array", freq_array.head(4))
    freq_array.to_csv("freq_array.csv")
    # filter the data according to a method specified in the parametres
    # we also save everything that is filtered out in df_unstable
    print("Filtering", flush=True)
    if options.filter == 'std':
        # we filter out everything where the score_std is higher than the threshold
        df_save = freq_array[freq_array.score_std < options.std_threshold]
        df_unstable = freq_array[freq_array.score_std >= options.std_threshold]
    if options.filter == 'selectf':
        print("l 235 selectr", flush=True)
        # here we look at the selection frequency e.g. how many separate sources the word is in
        # and drop if it is in less than the specified threshold (fraction)
        df_save = freq_array[freq_array['source'].apply(lambda x: len(x) >= options.selection_freq*num_files)]
        print("saving")
        df_save.to_csv("df_save.csv")
        #df_all = freq_array[freq_array['source'].apply(lambda x: len(x) >= 0)]
        df_unstable = freq_array[freq_array['source'].apply(lambda x: len(x) < options.selection_freq*num_files)]
        df_unstable.to_csv("df_unstable.csv")
    # sort the values by label and mean score, and take a certain amount of best results per label
    print("Sorting", flush=True)
    new_df_save = df_save.sort_values(['pred_label','score_mean'], ascending=[True, False]).groupby(['pred_label'],as_index=False).head(options.save_n)
    new_df_unstable = df_unstable.sort_values(['pred_label','score_mean'], ascending=[True, False]).groupby(['pred_label'],as_index=False).head(len(df_unstable))

    # we add a column that contains the number of sources
    # and drop the raw score and source columns
    new_df_save['source_number'] = new_df_save['source'].apply(lambda x: len(x))
    new_df_save.drop(['score','source'], axis=1, inplace=True)
#    new_df_save.to_csv("newdf_save.csv")
    #new_df_all['source_number'] = df_all['source'].apply(lambda x: len(x))
    #new_df_all.drop(['score','source'], axis=1, inplace=True)
    new_df_unstable['source_number'] = new_df_unstable['source'].apply(lambda x: len(x))
    new_df_unstable.drop(['score','source'], axis=1, inplace=True)
 #   new_df_unstable.to_csv("new_df_unstable.csv")
    #df_unstable.drop(['score'], axis=1, inplace=True)

    # removing pandas hierarchy for extracting labels
    # groupby() makes the (token, pred_label) pair be an index of the column, so it cant be easily referenced
    # take the index values (token, pred_label) and make them into a new frame
    new_df_save.to_csv("newdf_save.csv")
    new_df_unstable.to_csv("new_df_unstable.csv")


    df_sep = pd.DataFrame(data= new_df_save.index.values.tolist(), columns = ['token', 'pred_label'])
    # add all other data
    df_sep['score_mean'] = new_df_save['score_mean'].to_numpy()
    df_sep['score_std'] = new_df_save['score_std'].to_numpy()
    df_sep['source_number'] = new_df_save['source_number'].to_numpy()

    df_sep.to_csv("df_sep.csv")

    #print(df_sep)
    """for key in range(0,7):
        df_plot = df_sep[df_sep.pred_label == key]
        x = range(len(df_plot))
        y = df_plot['source_number']
        plt.plot(x,y)
        filename = options.plot_file+str(key)
        plt.savefig(filename)
    """

    # same for the other frame
    df_sep_unstable = pd.DataFrame(data= new_df_unstable.index.values.tolist(), columns = ['token', 'pred_label'])
    df_sep_unstable['score_mean'] = new_df_unstable['score_mean'].to_numpy()
    df_sep_unstable['score_std'] = new_df_unstable['score_std'].to_numpy()
    df_sep_unstable['source_number'] = new_df_unstable['source_number'].to_numpy()
    df_sep_unstable.to_csv("df_sep_unstable.csv")


    print("Saving unstable keywords...", flush=True)
    # save unstable words
    for key, number in {'HI': 0, 'ID':1, 'IN':2,'IP':3,'NA':4,'OP':5}.items():
        
        filename = options.unstable_file+key+".tsv"
        df_sep_unstable.to_csv(filename, sep="\t")

    # save results for all the labels separately
    print("Saving individuals with rare words removed", flush=True)
    filter_class_df(df_sep,"HI", 0, options)
    filter_class_df(df_sep,"ID", 1, options)
    filter_class_df(df_sep,"IN", 2, options)
    filter_class_df(df_sep,"IP", 3, options)
#    filter_class_df(df_sep,"LY", 4, options)
    filter_class_df(df_sep,"NA", 4, options)
    filter_class_df(df_sep,"OP", 5, options)
 #   filter_class_df(df_sep,"SP", 7, options)

    print("Everything done", flush=True)


if __name__=="__main__":
    #print("kws.py",flush = True)
    options = argparser().parse_args(sys.argv[1:])
    print(options,flush = True)
    process_data(options)
