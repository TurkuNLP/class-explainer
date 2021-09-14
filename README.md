# Class explainer for multilabel text classification

By Samuel Rönnqvist and Amanda Myntti

The goal of this project was to
- Extract keywords from multilabel data
- Analyse the quality of the extracted keywords

The extraction concists of the following parts:

- run_resplits.py, that runs train_multilabel.py and explain_multilabel.py
    - reads the data, preprocesses it and makes a new split into random training and validation sets.
    - trains the model on train data
    - predicts and calculates an attributions score with Integrated gradients method
run_resplits.py was run 100 times is our project

- run_evaluation.py, that runs kws.py and distinctiveness_and_coverage.py
    - reads the files produced by run_resplits.py
    - extracts keywords by looking at the attribution scores and their presence accross different runs
    - evaluates the resulting keywords using three different metrics


run_resplits.py requires following parameters:
- model_name (= 'xlm-roberta-base')
- data: path to unpreprosessed multilabel data
- batch_size, learning_rate, epochs, patience: for training 
- split, seed: ratio and random seed for splitting the training and validation sets (=0.67)
- checkpoints, save_model, save_explanations: directory for model checkpoints, filenames for model and the results

run_evaluation.py requires following parameters:
-
-
-
