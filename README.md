# Class explainer for multilabel text classification

By Samuel Rönnqvist and Amanda Myntti

The goal of this project was to
- Extract keywords from multilabel data
- Analyse the quality of the extracted keywords

For more information on the dataset and our motivation, please check: (link to wherever, the paper or documentation or...?)

## The extraction consists of the following parts:

- run_resplits.py, that runs train_multilabel.py and explain_multilabel.py
    - reads the data (train and dev sets), preprocesses it and makes a new split into random training and validation sets. 
    - trains the model on train data
    - predicts and calculates an attributions score with Integrated Gradients method
    - depending on the dataset size, running might take 10-20h
run_resplits.py was run 100 times in our project. 

- run_evaluation.py, that runs kws.py and distinctiveness_and_coverage.py
    - reads the files produced by run_resplits.py
    - extracts keywords by looking at the attribution scores and their presence accross different runs
    - evaluates the resulting keywords using three different metrics
    - running takes about 1-3h


## Parameters

run_resplits.py requires following parameters:
- *model_name*: name of the pretrained model. We used 'xlm-roberta-base' and the IG-aggregation score calculation is made around for XLM-RoBERTa's tokenization.
- *data*: path to unpreprosessed multilabel data.
- *batch_size*, *learning_rate*, *epochs*, *patience*: Parameters for training. We used batch_size = 30 , learning_rate = 7.5e-5 , epochs = 12 , and patience = 1.
- *split*, *seed*: ratio and random seed for splitting the training and validation sets. For split we had 0.67.
- *checkpoints*: directory for model checkpoint. We're saving *patience*+1 checkpoints.
- *save_model*: file for saving the model
- *save_explanations*: file for saving the results

The results are saved as a pandas dataframe:

```
document_id     true_label      pred_label      token       score
doc_0           [1,5]           1               words       0.122
doc_0           [1,5]           1               have        0.005
doc_0           [1,5]           1               meanings    0.093
doc_0           [1,5]           1               .           0.000
doc_0           [1,5]           4               words       0.044
doc_0           [1,5]           4               have        0.032
doc_0           [1,5]           4               meanings    0.009
doc_0           [1,5]           4               .           0.000
doc_1           [6]             6               It          0.002
...
...
```

run_evaluation.py requires following parameters:

- *data*: Path to the directory containing the explained (and simplified) files. The code reads all files that end in s.
- *choose_best*: The number of best words chosen per document-label pair at the start of extraction. We have used about 20. 
- *filter*: The type of filtering we do. There are two options: 'selectf' (selection frequency) and 'std' (standard deviation). We used 'selectf'.
- *fraction*: The selection frequency parameter for *filter'. We used 0.6 in our original paper.
- *threshold*: threshold for dropping words that have too low corpus frequency. We have used 5.
- *save_n*: Originally a parameter for saving *save_n* keywords per register, but evolved over time to limit the time for calculations, and cannot be removed for pandas-reasons. Keep this higher than the amount of keywords you need; e.g. for at least 100 keywords per register -> *choose_n* = 300.
- *save_file*: File name to save the resulting keywords to. The code adds the label name + .tsv to the file name. We also save the words that do not pass the corpus frequency filtering, those files are saved as *save_file* + label + err.tsv.
- *unstable_file*: File name for saving the words that do not pass the selection frequency filter. These are saved out of curiosity.
- *plot_file*: File name to save the plots. We plot the selection frequency of words against their rank on a list sorted by the IG aggregation scores. 
- *keyword_data*: Path to the keywords extracted before, thus is the same as *save_file*.
- *document_data*: Path to the document data, same as *data*. 
- *number_of_keywords*: the amount of best keywords chosen for analysis. We focused on top 100.
- *style*: ’TP’ = True Positive, ’TL’= True Label or ’P’ = Predictions. Tells us which labels to look at the coverage and corpus coverage phase. We used ’TP’ as the keywords have been extracted from true positives. ’TL’ and 'P' have been implemented out of curiosity.


## Example calls

```
srun python run_resplits.py \
  --data ./simplified-data/en \
  --model_name xlm-roberta-base \
  --lr 7.5e-5 \
  --epochs 12 \
  --batch_size 30 \
  --split 0.67 \
  --seed $2 \
  --patience 1 \
  --save_explanations explanations/$1 \
  --save_model models/$1 
```

```
srun python run_evaluation.py \
  --data ./class-explainer/explanations/run0 \
  --fraction 0.6 \
  --choose_best 20 \
  --save_n 500 \
  --threshold 5 \
  --save_file eval_output/kw_stable_all_$SLURM_JOBID \
  --unstable_file eval_output/kw_unstable_all_$SLURM_JOBID \
  --filter selectf \
  --keyword_data eval_output/kw_stable_all_$SLURM_JOBID \
  --document_data ./class-explainer/explanations/run0 \
  --plot_file eval_output/plot_$SLURM_JOBID \
  --results eval_output/result_$SLURM_JOBID

```
