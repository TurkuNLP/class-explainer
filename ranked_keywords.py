import pandas as pd
import numpy as np
from scipy.stats import rankdata
import glob




def read_data(data_name):
    """ read the data from a csv and remove null values (no predictions)"""

    data = pd.read_csv(data_name, delimiter = "\t", names = ['document_id', 'real_label', 'pred_label', 'token', 'score'])
    data['token'] = data['token'].str.lower()
    data['token'] = data['token'].fillna("NaN_")
    #data.dropna(axis = 0)

    return data


def choose_n_best(data, n):
    """ choose n best scoring words per document """

    df_topscores = pd.DataFrame(columns = ['document_id', 'real_label','pred_label','token', 'score'])#, 'logits']) 
    for doc in (set(data['document_id'])):
        df = data[data.document_id == doc]
        df_topscores = pd.concat([df_topscores,df.nlargest(n, 'score')], axis = 0)

    df_topscores.sort_index(inplace=True)

    return df_topscores


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




df_list = []

for filename in glob.glob("*.tsv"):
    print(filename)
    df = choose_n_best(read_data(filename),10)
    
    get_frequencies(df)
    df = drop_ambiguous_words(df, 3)
    
    rank(df)

    df_list.append(df)

# get all keywords
all_kws = []

for i in range(len(df_list)):
    all_kws.append(np.array(df_list[i]['token']))

all_kws = np.unique(np.array(all_kws).flatten()) 


df_full = pd.concat(df_list, ignore_index=True)

keywords = []



for word in all_kws:
    #print("in word ", word)
    counter = 0
    for i in range(len(df_list)):
        if word in np.array(df_list[i].token):
            counter += 1
    #print("counter is ", counter)
    if counter >= 0.8*len(df_list):
        df_sub = df_full[(df_full.token == word)]
        for label in set(df_sub['pred_label']):
           df_sub2 = df_sub[df_sub.pred_label == label]
           if len(df_sub2) >0:
              fre = len(df_sub2)
              classes = set(df_sub['pred_label'])
              mean = df_sub2['rank'].mean()
              std = df_sub2['rank'].std()
              min = df_sub2['rank'].min()
              max = df_sub2['rank'].max()
              keywords.append([label, word, fre, classes, mean, std, min, max]) 
        
df_comp = pd.DataFrame(data=keywords, columns = ['label','word', 'freq', 'class_freq', 'mean', 'std', 'min', 'max'])      

df_comp.sort_values(['label', 'mean'])



