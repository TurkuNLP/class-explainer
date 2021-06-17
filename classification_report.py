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
set_verbosity_error()    ## Makes the progress bars go away



fname = 'eacl_sv_binarized.pkl'
modelname = 'enfifr'

# Hyperparameters
LEARNING_RATE=1e-5
BATCH_SIZE=4
TRAIN_EPOCHS=5
MODEL_NAME = 'xlm-roberta-base'
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# these are printed out by make_dataset.py 
labels = ['HI', 'ID', 'IN', 'IP', 'LY', 'NA', 'OP', 'SP']



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


def read_dataset():
  """
  Read the data. Labels should be in the form of binary vectors.
  """
  

  with open('binarized_data/'+fname, 'rb') as f:
    dataset = pickle.load(f)
  print("Dataset succesfully loaded")
  print(dataset)
  return dataset

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

    y_pred = pred.predictions
    y_true = pred.label_ids.reshape(y_pred.shape)
    

    # apply sigmoid
    y_pred_s = 1.0/(1.0 + np.exp(-y_pred))

    threshold = 0.5  
    pred_ones = [pl>threshold for pl in y_pred_s]
    true_ones = [tl==1 for tl in y_true]
    return { 'accuracy': classification_report(true_ones,pred_ones, target_names = labels) }



dataset = read_dataset()
model = torch.load('models/multilabel_model_'+modelname+'.pt')
print("Evaluating "+fname+" on "+modelname)

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
      disable_tqdm=True
  )


trainer = MultilabelTrainer(
      model,
      train_args,
      train_dataset=encoded_dataset['train'],
      eval_dataset=encoded_dataset['test'],
      tokenizer=tokenizer,
      compute_metrics=compute_accuracy
  )


# If you wonder why I did this this way and not just model.predict()
# or something like that,
# this was completely copy-pasteable :D

results = trainer.evaluate()
print(results['eval_accuracy'])
