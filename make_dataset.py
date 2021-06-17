from datasets import load_dataset
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
import pickle


# remove all progress bars
from datasets.utils.logging import set_verbosity_error
set_verbosity_error()  



# parametres
tr = 'en'  # binarizing this
unwanted_labels = ['DF', 'DT', 'OS', 'PO', 'QA']
sub_labels = ['DS', 'IT', 'MT', 'NE', 'EN']

def remove_NA(d):
  """
  Remove null values and separate multilabel values with comma
  """
  if d['label'] == None:
    d['label'] = np.array('NA')
  if ' ' in d['label']:
    d['label'] = ",".join(sorted(d['label'].split()))
  return d

def label_encoding(d):
  """
  Split the multi-labels
  """
  d['label'] = np.array(d['label'].split(","))
  return d

def remove_falselabels(d):
  d['label'] = [label for label in d['label'] if label not in unwanted_labels]
  for i in range(len(d['label'])):
    if d['label'][i] in sub_labels:
      d['label'][i] = str.lower(d['label'][i])
  return d

def remove_sublabels(d):
  """
  Remove the sub-labels (lower case) from data.
  """
  d['label'] =  [label for label in d['label'] if label.isupper()]
  return d


def make_dataset():

    PATH = '../../veronika/simplified-data/'

    # read the data
    dataset_0 = load_dataset(
        'csv', 
        data_files={
        'train': [PATH+tr+'/train.tsv-simp.tsv'], 
        'validation': [PATH+tr+'/dev.tsv-simp.tsv'], 
        'test': [PATH+tr+'/test.tsv-simp.tsv']
        },
        delimiter='\t', 
        column_names=['label', 'sentence']
        )

    # read all train data to ensure similar binarisation on all data 
    dataset_fit = load_dataset('csv', data_files={
        'train' : [PATH+'en/train.tsv-simp.tsv',PATH+'fi/train.tsv-simp.tsv',
                  PATH+'fr/train.tsv-simp.tsv',PATH+'sv/train.tsv-simp.tsv']},
        delimiter='\t', column_names=['label','sentence'])

    print("Removing null values:")
    dataset = dataset_0.map(remove_NA)
    dataset_fit = dataset_fit.map(remove_NA)

    print("Separating multilabels:")
    dataset = dataset.map(label_encoding)
    dataset_fit = dataset_fit.map(label_encoding)

    print("Removing unwanted labels:")
    dataset = dataset.map(remove_falselabels)
    dataset_fit = dataset_fit.map(remove_falselabels)
    
    print("Check:")
    print(dataset['train']['label'][0])
    print(dataset['test']['label'][0])

    print("Removing sub-labels from evaluation data")
    dataset['validation'] = dataset['validation'].map(remove_sublabels)
    dataset['test'] = dataset['test'].map(remove_sublabels)
    
    print("Check:")
    print(dataset['train']['label'][0])
    print(dataset['test']['label'][0])

    mlb = MultiLabelBinarizer()
  
    # fit mlb with the fit data
    onehot_train = mlb.fit(dataset_fit['train']['label'])
    labels = mlb.classes_
    print("Labels of the data:", labels)

    print("Binarizing the labels:")
    dataset = dataset.map(lambda line: {'label': mlb.transform([line['label']])})

    # save it
    with open("binarized_data/eacl_"+tr+"_binarized.pkl", 'wb') as f:
        pickle.dump(dataset,f)


if __name__=="__main__":
    print("Binarizing ",tr)
    make_dataset()

