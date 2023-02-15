from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments
from transformers import Trainer
from transformers import EarlyStoppingCallback
from datasets import load_dataset
from datasets import concatenate_datasets
import datasets
import pickle
import sys
import re
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from torch import cuda
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.preprocessing import MultiLabelBinarizer
import collections

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import sys
import random

from transformers.utils.logging import DEBUG as log_level
from transformers.utils.logging import set_verbosity as model_verbosity
from datasets.utils.logging import set_verbosity as data_verbosity
model_verbosity(log_level)
data_verbosity(log_level)
#set_verbosity_error()    ## lisäys


# Hyperparameters
tr =  ['en', 'fi', 'fr']
LEARNING_RATE=1e-4
BATCH_SIZE=30
TRAIN_EPOCHS=2
MODEL_NAME = 'xlm-roberta-base'
MAX_DEVIANCE = 0.005
PATIENCE = 5
# these are printed out by make_dataset.py
labels = ['HI', 'ID', 'IN', 'IP', 'NA', 'OP']#, 'av', 'ds', 'dtp', 'ed', 'en', 'fi', 'it', 'lt', 'mt', 'nb', 'ne', 'ob', 'ra', 're', 'rs', 'rv', 'sr']


def argparser():
    ap = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    ap.add_argument('--model_name', default=MODEL_NAME,
                    help='Pretrained model name')
    ap.add_argument('--data', metavar='FILE', required=True,
                    help='Path to binarized dataset')
    ap.add_argument('--batch_size', metavar='INT', type=int,
                    default=BATCH_SIZE,
                    help='Batch size for training')
    ap.add_argument('--epochs', metavar='INT', type=int, default=TRAIN_EPOCHS,
                    help='Number of training epochs')
    ap.add_argument('--lr', '--learning_rate', metavar='FLOAT', type=float,
                    default=LEARNING_RATE, help='Learning rate')
    ap.add_argument('--patience', metavar='INT', type=int,
                    default=PATIENCE, help='Early stopping patience')
    ap.add_argument('--split', metavar='FLOAT', type=float,
                    default=None, help='Set fixed train/val data split ratio (e.g. 0.8 to train on 80%)')
    ap.add_argument('--seed', metavar='INT', type=int,
                    default=None, help='Random seed for splitting data')
    ap.add_argument('--checkpoints', default=None, metavar='FILE',
                    help='Save model checkpoints to directory')
    ap.add_argument('--save_model', default=None, metavar='FILE',
                    help='Save model to file')
    #ap.add_argument('--save_predictions', default=False, action='store_true',
    #                help='save predictions and labels for dev set, or for test set if provided')
    return ap



# overriding the loss-function in Trainer
class MultilabelTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.BCEWithLogitsLoss()
        loss = loss_fct(logits.view(-1, self.model.config.num_labels),
                        labels.float().view(-1, self.model.config.num_labels))
        return (loss, outputs) if return_outputs else loss


def get_label_counts(dataset):
    """ Calculates the frequencies of labels of a dataset. """
    label_counts = collections.Counter()
    for line in dataset:
        for label in line['label']:
            label_counts[label] += 1
    return label_counts


def resplit2(dataset):
    # this is the one used as June 2022
# there are multilingual train+val sets + lang specific val sets
    dataset_en = dataset["en"].train_test_split(test_size=0.5,shuffle=True)
    dataset_fi = dataset["fi"].train_test_split(test_size=0.5,shuffle=True)
    dataset_fr = dataset["fr"].train_test_split(test_size=0.5,shuffle=True)
    dataset["train"] = concatenate_datasets([dataset_en['train'], dataset_fi['train'],dataset_fr['train']])
    dataset["validation"] = concatenate_datasets([dataset_en['test'], dataset_fi['test'],dataset_fr['test']])
    dataset["validation_en"] = dataset_en["test"]
    dataset["validation_fr"] = dataset_fr["test"]
    dataset["validation_fi"] = dataset_fi["test"]
    return dataset

def resplit(dataset, ratio=0.5, seed=None): # not used
    """ Shuffle and resplit train and validation sets """

    # get all labels for comparison
    all_label_counts = get_label_counts(dataset['concat'])

    if seed is not None:
        random.seed(seed)

    original_train_size = len(dataset['train'])
    while True:
        # We shuffle the save data in concat
        #dataset['concat'].shuffle(seed=seed)
        new_order = list(range(len(dataset['concat'])))
        random.shuffle(new_order)
        # Three-way confirmation of whether the split is good (all labels in train, all labels in val, values balanced)
        ok = [False]*3
        deviance = 0
        # i = index in ok, key = {train,validation}, beg = start index, end = end index
        for i, (key,beg,end) in enumerate([('train', 0, int(len(dataset['concat'])*ratio)), ('validation', int(len(dataset['concat'])*ratio), len(dataset['concat']))]):
            # insert ratioed amount of dataset[concat] to key
            #dataset[key] = dataset['concat'].select(range(beg,end))
            dataset[key] = dataset['concat'].select(new_order[beg:end])
            # get labels and see if all labels are represented, if yes: ok[i] = True
            label_counts = get_label_counts(dataset[key])
            ok[i] = len(label_counts) == len(all_label_counts)
            # Check deviance, add them up
            for label, count in all_label_counts.most_common():
                deviance += float(label_counts[label]/len(dataset[key])-count/len(dataset['concat']))
        # See if deviance/set is good
        if deviance/2.0 <= MAX_DEVIANCE:
            ok[-1] = True

        # if all is good
        if all(ok):
          print("Split successfull! Deviance: ",deviance)
          print("Label distribution: ")
          print("train: ", get_label_counts(dataset['train']))
          print("val: ", get_label_counts(dataset['validation']))
          data_indexes = [(i//original_train_size, i%original_train_size) for i in new_order]
          open("data_split_%s.json" % seed, 'w').write("{\n'train': %s\n,\n'val': %s\n}" % (str(data_indexes[:int(len(dataset['concat'])*ratio)]), str(data_indexes[int(len(dataset['concat'])*ratio):])))
          return dataset, data_indexes[:int(len(dataset['concat'])*ratio)], data_indexes[int(len(dataset['concat'])*ratio):] 
        else:
          print("Split unsuccessfull.")
          if ratio>0.99:
            print("Disregarding validation distribution because of intentionally skewed split ratio.")
            data_indexes = [(i//original_train_size, i%original_train_size) for i in new_order]
            open("data_split_%s.json" % seed, 'w').write("{\n'train': %s\n,\n'val': %s\n}" % (str(data_indexes[:int(len(dataset['concat'])*ratio)]), str(data_indexes[int(len(dataset['concat'])*ratio):])))
          #  print("XXX dataset", dataset, "indexes", data_indexes[:int(len(dataset['concat'])*ratio)])
            return dataset, data_indexes[:int(len(dataset['concat'])*ratio)], data_indexes[int(len(dataset['concat'])*ratio):] 




def sample_dataset(d,n,sample):
  """
  Sample = {'up', 'down'}.
  If up, upsample the data by 100 persent.
  If down, downsample the data to 1/n percent.
  """

  if sample == 'up':
    d_shuffled_train = d['train'].shuffle(seed = 123)
    d_shuffled_validation = d['validation'].shuffle(seed = 123)
    d_shuffled_test = d['test'].shuffle(seed = 123)

    d['train'] = datasets.concatenate_datasets([d['train'], d_shuffled_train])
    d['validation'] = datasets.concatenate_datasets([d['validation'], d_shuffled_validation])
    d['test'] = datasets.concatenate_datasets([d['test'], d_shuffled_test])

    return d

  if sample == 'down':
    d['train'] = d['train'].shuffle(seed = 123).shard(n,0)
    d['validation'] = d['validation'].shuffle(seed = 123).shard(n,0)
    d['test'] = d['test'].shuffle(seed = 123).shard(n,0)

    return d


def preprocess_text(d):
    # Separate punctuations from words by whitespace
    try:
        d['sentence'] = re.sub(r"([\.,:;\!\?\"\(\)])([\w\d])", r"\1 \2", re.sub(r"([\.,:;\!\?\"\(\)])([\w\d])", r"\1 \2", d['sentence']))
    except:
        print("Warning: Unable to run regex on text of type", type(d['sentence']))
    return d

def remove_NA(d):
  """ Remove null values and separate multilabel values with comma """
  if d['label'] == None:
    d['label'] = np.array('NA')
  if ' ' in d['label']:
    d['label'] = ",".join(sorted(d['label'].split()))
  return d


def label_encoding(d):
  """ Split the multi-labels """
  d['label'] = np.array(d['label'].split(","))
  return d


def remove_sublabels(d):
  """ Remove the sub-labels (lower case) from data. Used for validation. """
  d['label'] =  [label for label in d['label'] if label.isupper()]
  return d


def binarize(dataset):
    """ Binarize the labels of the data. Fitting based on the whole data. """
    mlb = MultiLabelBinarizer()
    mlb.fit([labels])
    print("Binarizing the labels")
    dataset = dataset.map(lambda line: {'label': mlb.transform([line['label']])})
    return dataset


def read_dataset(path):
  data_files = path
  print("XXX data_files", path)
  dataset = load_dataset(
        'csv',
        data_files=data_files,
        delimiter='\t',
        column_names=['label', 'sentence'],
        cache_dir = "v_cachedir"
        )

  # Remove errors and format the labels
  dataset = dataset.map(remove_NA)
  dataset = dataset.map(label_encoding)
  dataset = dataset.map(preprocess_text)

  # get the label distribution of the whole data
 # label_counts = get_label_counts(dataset['concat'])
  #print("Labels of the whole dataset: ",label_counts)

  print("Dataset succesfully loaded. ")
  return dataset



def wrap_tokenizer_fn(tokenizer):
    def encode_dataset(d):
      """
      Tokenize the sentences. Null/None sentences converted to empty strings.
      """
      try:
        output = tokenizer(d['sentence'], truncation= True, padding = True, max_length=512)
        return output
      except:     #there were a few empty sentences
        output = tokenizer(" ", truncation= True, padding = True, max_length=512)
        return output

    return encode_dataset


def compute_accuracy(pred):
    # flatten them to 1D vectors
    y_pred = pred.predictions.flatten()
    y_true = pred.label_ids.flatten()

    # apply sigmoid
    y_pred_s = 1.0/(1.0 + np.exp(-y_pred))

    threshold = 0.5
    pred_ones = [pl>threshold for pl in y_pred_s]
    true_ones = [tl==1 for tl in y_true]
    return { 'accuracy': f1_score(y_true=true_ones, y_pred=pred_ones, average='weighted') }
    #return { 'accuracy': accuracy_score(true_ones, pred_ones) }



def train(dataset, options):

  # print the hyperparametres
  print("Model type: ", options.model_name)
  print("Learning rate: ", options.lr)
  print("Batch size: ", options.batch_size)
  print("Epochs: ", options.epochs)
  # Model downloading
  num_labels = len(dataset['train']['label'][0][0]) #here double brackets are needed!
  print("Downloading model", flush=True)
  model = AutoModelForSequenceClassification.from_pretrained(options.model_name, num_labels = num_labels)
  tokenizer = AutoTokenizer.from_pretrained(options.model_name)

  print("Tokenizing data", flush=True)
  encoded_dataset = dataset.map(wrap_tokenizer_fn(tokenizer))

  print("Initializing model", flush=True)
  train_args = TrainingArguments(
        options.save_model+'-ckpt',    # output directory for checkpoints and predictions
        load_best_model_at_end=True,
        evaluation_strategy='epoch',
        logging_strategy='epoch',
        learning_rate=options.lr,
        per_device_train_batch_size=options.batch_size,
        per_device_eval_batch_size=options.batch_size,
        num_train_epochs=options.epochs,
        gradient_accumulation_steps=4,
        save_total_limit=options.patience+1,
        disable_tqdm=False   ## True=disable progress bar in training
    )


  trainer = MultilabelTrainer(
        model,
        train_args,
        train_dataset=encoded_dataset['train'],
        eval_dataset=encoded_dataset['validation'],
        tokenizer=tokenizer,
        compute_metrics=compute_accuracy,
        callbacks = [EarlyStoppingCallback(early_stopping_patience=options.patience)]
    )

  print("Training", flush=True)
  if options.epochs > 0:
      trainer.train()
  # Evaluate
  print("Evaluating", flush=True)
  #results = trainer.evaluate()
  #print('Accuracy:', results["eval_accuracy"])

  # for classification report: get predictions
  val_predictions = trainer.predict(encoded_dataset['validation'])

  # apply sigmoid to predictions and reshape real labels
  p = 1.0/(1.0 + np.exp(- val_predictions.predictions))
  t = val_predictions.label_ids.reshape(p.shape)
  # apply treshold of 0.5
  pred_ones = [pl>0.5 for pl in p]
  true_ones = [tl==1 for tl in t]

  print(classification_report(true_ones,pred_ones, target_names = labels))
  report = classification_report(true_ones,pred_ones, target_names = labels, output_dict=True)

  # save the model
  if options.save_model is not None:
     torch.save(trainer.model, options.save_model)#"models/multilabel_model3_fifrsv.pt")

  return model, tokenizer, report



if __name__=="__main__":
  options = argparser().parse_args(sys.argv[1:])
  device = 'cuda' if cuda.is_available() else 'cpu'

  print("Reading data")
  dataset = read_dataset(options.data)
  print("Splitting data")
  dataset = resplit2(dataset)#, ratio=options.split, seed=options.seed)
  print("Binarizing data")
  dataset = binarize(dataset)
  print("Ready to train:")
  train(dataset, options)
