# Class explainer for multilabel text classification

This repository contains the code for the *Stable Attribution Class Explanation (SACX)* method for providing explanations of text classes in the form of keyword lists, based on input attribution (Integrated Gradients) from deep text classification models (e.g. XLM-R), and repeated training/explaining in order to stabilize the keywords. The code is developed by Samuel Rönnqvist and Amanda Myntti, and the original explainability code by Filip Ginter. The method and its systematic evaluation is described in the paper [*Explaining Classes through Stable Word Attributions*](https://aclanthology.org/2022.findings-acl.85/) by Samuel Rönnqvist, Amanda Myntti, Aki-Juhani Kyröläinen, Filip Ginter and Veronika Laippala. Please cite, if you make use of this code.

```
@inproceedings{ronnqvist-etal-2022-explaining,
    title = "Explaining Classes through Stable Word Attributions",
    author = {R{\"o}nnqvist, Samuel and Kyr{\"o}l{\"a}inen, Aki-Juhani and Myntti, Amanda and Ginter, Filip and Laippala, Veronika},
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2022",
    year = "2022",
    pages = "1063--1074",
}
```

### Quick start (with slurm)
Training the classifier and explaining on a validation split, e.g. 20 times: 
```
for i in {000..019}; do sbatch sl-train-explain.bash run$i 123$i; done
```
(saving explanation files as run000 etc., using seed 123000 etc.)

Converting explanation files:
```
for N in {000..019}; do python3 convert_explanations.py explanations/run$N; done
```

Extracting keywords:
```
sbatch sl-eval.bash 0.7 20 0.7 0.3 run0
```

### Alternative: multilingual setup
```
for i in {000..019}; do sbatch sl-train-explain-multiling.bash multiling$i 123$i;done
for N in {000..019}; do for LANG in ar en fi fr zh; do python3 convert_explanations_multi.py explanations/multiling${N}p_$LANG.tsv explanations/multiling${N}a_$LANG.tsv explanations/multiling-$LANG-${N}; done; done
python3 count_class_words_multiling.py
for LANG in ar en fi fr zh; do sbatch sl-eval-multiling.bash 0.7 20 0.7 0.3 multiling-$LANG; done
```

## Components

The extraction consists of the following parts:

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

The results are saved in two types of TSV files: prediction files (e.g. run000p.tsv) and word attribution files (e.g. run000a.tsv). The <code>convert_explanations.py</code> script combines both and produces two other types of files used for the subsequent processing: score files (e.g. run000s.tsv) and text files (e.g. run000wNF.tsv).

<code>run_evaluation.py</code> extracts keywords and performs evaluation of them. It uses a document frequency dictionary (class_df.json) that can be pre-calculated using <code>count_class_words.py</code>.

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
