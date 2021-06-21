from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments
from transformers import Trainer
from datasets import load_dataset
import datasets
import pickle
import sys
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from torch import cuda
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import sys
import random

from datasets.utils.logging import set_verbosity_error
set_verbosity_error()    ## lisäys

# Hyperparameters
tr =  ['en', 'fi', 'fr', 'sv']
LEARNING_RATE=1e-5
BATCH_SIZE=16
TRAIN_EPOCHS=6
MODEL_NAME = 'xlm-roberta-base'
# these are printed out by make_dataset.py
labels = ['HI', 'ID', 'IN', 'IP', 'LY', 'NA', 'OP', 'SP', 'av', 'ds', 'dtp', 'ed', 'en', 'fi', 'it', 'lt', 'mt', 'nb', 'ne', 'ob', 'ra', 're', 'rs', 'rv', 'sr']


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
    ap.add_argument('--split', metavar='FLOAT', type=float,
                    default=None, help='Set fixed train/val data split ratio')
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


def read_dataset(path):
  """
  Read the data. Labels should be in the form of binary vectors.
  """
  #data_name = 'binarized_data/eacl_'+tr+'_binarized.pkl'
  with open(path, 'rb') as f:
    dataset = pickle.load(f)
  """if tr in ['fr','sv']:
    dataset = sample_dataset(dataset,2, 'up')
    dataset = sample_dataset(dataset,2, 'up')
  if tr == 'en':
    dataset = sample_dataset(dataset,5,'down')"""
  print("Dataset succesfully loaded: "+data_name)
  return dataset

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
    return { 'accuracy': accuracy_score(pred_ones, true_ones) }



def train(dataset, options):

  # print the hyperparametres
  print("Model type: ", options.model_name)
  print("Learning rate: ", options.lr)
  print("Batch size: ", options.batch_size)
  print("Epochs: ", options.epochs)

  # Model downloading
  num_labels = len(dataset['train']['label'][0][0]) #here double brackets are needed!
  print("Downloading model")
  model = AutoModelForSequenceClassification.from_pretrained(options.model_name, num_labels = num_labels)
  tokenizer = AutoTokenizer.from_pretrained(options.model_name)

  print("Tokenizing data")
  encoded_dataset = dataset.map(wrap_tokenizer_fn(tokenizer))

  train_args = TrainingArguments(
        'multilabel_model_checkpoints',    # output directory for checkpoints and predictions
        load_best_model_at_end=True,
        evaluation_strategy='epoch',
        logging_strategy='epoch',
        learning_rate=options.lr,
        per_device_train_batch_size=options.batch_size,
        num_train_epochs=options.epochs,
        gradient_accumulation_steps=4,
        save_total_limit=3,
        disable_tqdm=True   ## lisäys
    )


  trainer = MultilabelTrainer(
        model,
        train_args,
        train_dataset=encoded_dataset['train'],
        eval_dataset=encoded_dataset['validation'],
        tokenizer=tokenizer,
        compute_metrics=compute_accuracy
    )

  print("Ready to train")
  trainer.train()

  results = trainer.evaluate()

  print('Accuracy:', results["eval_accuracy"])

  # for classification report:
  val_predictions = trainer.predict(encoded_dataset['validation'])

  # apply sigmoid to predictions and reshape real labels
  p = 1.0/(1.0 + np.exp(- val_predictions.predictions))
  t = val_predictions.label_ids.reshape(p.shape)

  pred_ones = [pl>0.5 for pl in p]
  true_ones = [tl==1 for tl in t]

  # labels are printed by make_dataset.py copy them to hyperparametres
  print(classification_report(true_ones,pred_ones, target_names = labels))

  # save the model
  if options.save_model is not None:
      torch.save(trainer.model, options.save_model)#"models/multilabel_model3_fifrsv.pt")


def resplit(dataset, ratio=None, seed=None):
    """ Shuffle and resplit train and validation sets """

    if seed is not None:
        random.seed(seed)

    if ratio is None:
        #TODO: check train/val sizes from data and use same ratio
        pass
    else:
        #TODO: use given ratio
        pass


if __name__=="__main__":
  options = argparser().parse_args(sys.argv[1:])
  device = 'cuda' if cuda.is_available() else 'cpu'

  # This part left out as long as we do monolingual training
  """dataset = read_dataset(tr[0])
  for i in range(1,len(tr)):
    d = read_dataset(tr[i])
    dataset['train'] = datasets.concatenate_datasets([dataset['train'], d['train']])
    dataset['validation'] = datasets.concatenate_datasets([dataset['validation'], d['validation']])
    dataset['test'] = datasets.concatenate_datasets([dataset['test'], d['test']])"""

  dataset = read_dataset(options.data)
  dataset = resplit(dataset, ratio=0.5, seed=options.seed)
  train(dataset, options)
