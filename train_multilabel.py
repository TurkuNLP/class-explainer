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

from datasets.utils.logging import set_verbosity_error
set_verbosity_error()    ## lisäys

# Hyperparameters
tr =  ['fi', 'fr', 'sv']
LEARNING_RATE=1e-5
BATCH_SIZE=16
TRAIN_EPOCHS=6
MODEL_NAME = 'xlm-roberta-base'
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# these are printed out by make_dataset.py 
labels = ['HI', 'ID', 'IN', 'IP', 'LY', 'NA', 'OP', 'SP', 'av', 'ds', 'dtp', 'ed', 'en', 'fi', 'it', 'lt', 'mt', 'nb', 'ne', 'ob', 'ra', 're', 'rs', 'rv', 'sr']


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


def read_dataset(tr):
  """
  Read the data. Labels should be in the form of binary vectors.
  """
  data_name = 'binarized_data/eacl_'+tr+'_binarized.pkl'
  with open(data_name, 'rb') as f:
    dataset = pickle.load(f)
  if tr in ['fr','sv']:
    dataset = sample_dataset(dataset,2, 'up')
    dataset = sample_dataset(dataset,2, 'up')
  if tr == 'en':
    dataset = sample_dataset(dataset,5,'down')
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



def train(dataset):

  # print the hyperparametres
  print("Learning rate: ", LEARNING_RATE)
  print("Batch size: ", BATCH_SIZE)
  print("Epochs: ", TRAIN_EPOCHS)
  
  # Model downloading
  num_labels = len(dataset['train']['label'][0][0]) #here double brackets are needed!
  print("Downloading model")
  model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels = num_labels)
  
  print("Tokenizing data")
  encoded_dataset = dataset.map(encode_dataset)
  
  train_args = TrainingArguments(
        'multilabel_model_checkpoints',    # output directory for checkpoints and predictions
        load_best_model_at_end=True,
        evaluation_strategy='epoch',
        logging_strategy='epoch',
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        num_train_epochs=TRAIN_EPOCHS,
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
  torch.save(trainer.model,"models/multilabel_model3_fifrsv.pt")

if __name__=="__main__":
  
  device = 'cuda' if cuda.is_available() else 'cpu'
  dataset = read_dataset(tr[0])

  for i in range(1,len(tr)):
    d = read_dataset(tr[i])
    dataset['train'] = datasets.concatenate_datasets([dataset['train'], d['train']])
    dataset['validation'] = datasets.concatenate_datasets([dataset['validation'], d['validation']])
    dataset['test'] = datasets.concatenate_datasets([dataset['test'], d['test']])
  train(dataset)
