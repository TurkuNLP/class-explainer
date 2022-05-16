import distinctiveness_and_coverage as dist_and_cov_works
import kws
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import sys
from numpy.core.numeric import NaN
import warnings
import json
import numpy as np
import pandas as pd
import re
import glob
import time
from collections import Counter
import collections
import matplotlib.pyplot as plt

WORDS_PER_DOC = 10
SELECTION_FREQ = 0.6
MIN_WORD_FREQ = 5
SAVE_N = 1000
SAVE_FILE = 'eval_output/first-run-'
UNSTABLE_FILE = 'eval_output/unstables/first-run-unst-'
NUMBER_OF_KEYWORDS = 100
PREDICTION_THRESHOLD = 0.5
FREQUENT_PREDICTIONS_THRESHOLD = 0.5




def argparser():
    ap = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    ap.add_argument('--data', metavar='FILE', required=True,
                    help='Path to data scored data')
    ap.add_argument('--words_per_doc', metavar='INT', type=int, default=WORDS_PER_DOC,
                    help='Number of best words chosen per each doc-label pair. Optimize')
    ap.add_argument('--filter', metavar='FILE', default = 'selectf',
                    help='method for filtering, std or selectf')
    ap.add_argument('--selection_freq', metavar='FLOAT', type=float,
                    default=SELECTION_FREQ, help='% in how many lists the word must be present in (selection frequency). Optimize')
    ap.add_argument('--min_word_freq', metavar='INT', type=int, default=MIN_WORD_FREQ,
                    help='Threshold for dropping words that are too rare')
    ap.add_argument('--save_n', metavar='INT', type=int, default=SAVE_N,
                    help='How many words per class are saved. This needs to be really high, explanation in comments')
    ap.add_argument('--save_file', default=SAVE_FILE, metavar='FILE',
                    help='dir+file for saving the the keywords')
    ap.add_argument('--unstable_file', default=UNSTABLE_FILE, metavar='FILE',
                    help='dir+file for saving the unstable results')
    ap.add_argument('--keyword_data', metavar='FILE', required=True,
                    help='Path to keyword data. SAME AS save_file')
    ap.add_argument('--document_data', metavar='FILE', required=True,
                    help='Path to document data, ALL DOCS TEXTS AND PREDICTIONS')
    ap.add_argument('--number_of_keywords', metavar='INT', type=int, default=NUMBER_OF_KEYWORDS,
                    help='Threshold for number of keywords compared/chosen per register. FIXED 100')
    ap.add_argument('--style', metavar='STR', type=str, default='TP',
                    help='TP = True Positive, P = Predictions, TL = True Label')
    ap.add_argument('--prediction_th', type=float, default=PREDICTION_THRESHOLD,
                    help='Threshold on model posterior probability. For logging purposes only, threshold is controlled by --data.')
    ap.add_argument('--frequent_predictions_th', type=float, default=FREQUENT_PREDICTIONS_THRESHOLD,
                    help='Threshold for choosing best predictions from all predcitions. Maybe optimizable?')

    ap.add_argument('--plot_file', metavar='FILE', required=True,
                    help='File to save plots')
    ap.add_argument('--class_df', metavar='FILE', required=False, default='class_df.json',
                    help='Class DF file.')
    #ap.add_argument('--results', metavar='FILE', required=True,
    #                help='File to save final results')
    return ap


if __name__ == '__main__':
    options = argparser().parse_args(sys.argv[1:])
    print(options,flush = True)

    # kws.py -part:
    # Do calculations and save them to --save_file

    kws.process_data(options)
    #dist_and_cov_works.calculate(options)

    dist_and_cov_works.evaluate_keywords_file(options)
