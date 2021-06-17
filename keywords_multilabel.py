import pandas as pd
import numpy as np

# really bad code tbh


def write_kw(df, multilabels = None):
    """"
    basic file writing but with many many conditions
    If you want to write about documents with multilabels, provide the set of
    those label pairs in multilabels
    """
    if multilabels == None:
        for i in set(df['pred_label']):
            filename = 'zeroshot2/kw_results_sv/kw_label_'+str(i)+'.txt'
            f = open(filename, 'w')
            f.write("Class "+str(i)+" ("+labels[int(i)]+")\n")
            kw = df[(df.freq >= 3) & (df.pred_label == i) & (df.class_freq < 4)]
            kw.sort_values(['score'], ascending=False, inplace=True)
            kw.drop_duplicates(subset='token', keep="first", inplace=True)
            # write 100 first words
            for token in kw['token'][:100]:
                f.write(token+"\n")
            f.write("\n")
            f.close()
    else:
        for (i,j) in multilabels:
            filename = 'kw_results/kw_labels_'+str(i)+str(j)+'.txt'
            f = open(filename, 'w')
            f.write("Class "+str(i)+"+"+str(j)+" ("+labels[int(i)]+"+"+labels[int(j)]+") \n")
            kw = df[(df.freq >= 2) & (df.class_freq < 4) & (df.multilabel == {i,j})]
            kw.sort_values(['score'], ascending=False, inplace=True)
            kw.drop_duplicates(subset='token', keep="first", inplace=True)
            # write 100 first words
            for token in kw['token'][:100]:
                f.write(token+"\n")
            f.write("\n")
            f.close()
            
    
# these given by make_dataset, remember to copy here
labels = ['HI', 'ID', 'IN', 'IP', 'LY', 'NA', 'OP', 'SP']

# read the data that has already aggregation scores calculated (explain_multilabel.py)
data = pd.read_csv('zeroshot2/zeroshot_sv_scores.tsv', sep = '\t', names = ['document_id', 'pred_label', 'token', 'score'])
# lowercase and remove None values
data['token'] = data['token'].str.lower()
data['token'] = data['token'].fillna("NaN_")


# a df for best words
df_topscores = pd.DataFrame(columns = ['document_id', 'pred_label', 'token', 'score'])

# this concatenates too much but works. For each doc (and label in doc)
# take the 10 best scoring words
for doc in (set(data['document_id'])):
    df = data[data.document_id == doc]
    for label in set(df['pred_label']):
        df_topscores = pd.concat([df_topscores,df[df.pred_label == label].nlargest(10, 'score')], axis = 0)

# sort for easy readability
df_topscores.sort_index(inplace=True)

# see how it looks
#print(df_topscores[df_topscores.document_id == 'document_4'])

# calculate frequencies so that we can eliminate noise
freq_df = df_topscores.groupby('token').count()

# add the frequencies to df_topscores
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

# find the frequency of classes where the word is used often 
# to eliminate "to", "and", etc. later
label_numbers = []
for index, row in df_topscores.iterrows():
    classes_list = list(df_topscores[df_topscores.token == row['token']]['pred_label'])
    label_numbers.append(len(set(classes_list)))

df_topscores['class_freq'] = label_numbers



# divide to mono and multilabels (was not actually that usefull)
multilabels = []
mul_labels_for_df = []

df_mono = pd.DataFrame(columns = ['document_id', 'pred_label', 'token', 'score', 'freq', 'class_freq'])
df_multi = pd.DataFrame(columns = ['document_id', 'pred_label', 'token', 'score', 'freq', 'class_freq'])


for doc in (set(data['document_id'])):
    df = df_topscores[df_topscores.document_id ==doc]
    a = set(df['pred_label'])
    if len(a) >1:
        l = len(df)
        df_multi = pd.concat([df_multi,df], axis = 0)
        for i in range(l):
            mul_labels_for_df.append(a)
        if a not in multilabels:
            multilabels.append(a)
    else:
        df_mono = pd.concat([df_mono,df], axis = 0)
   
df_multi['multilabel'] = mul_labels_for_df
df_mono.sort_index(inplace=True)
df_multi.sort_index(inplace=True)


# write separately for multi and monolabels
#write_kw(df_mono)        
#write_kw(df_multi, multilabels = multilabels)

# if that kind of split is not needed, just
write_kw(df_topscores)

    
