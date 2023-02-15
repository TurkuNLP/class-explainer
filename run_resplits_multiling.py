import train_multilabel
import explain_multilabel
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import sys
import numpy as np
import pandas as pd
import json

LEARNING_RATE=1e-4
BATCH_SIZE=30
TRAIN_EPOCHS=2
MODEL_NAME = 'xlm-roberta-base'
MAX_DEVIANCE = 0.005
PATIENCE = 3


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
    default=None, help='Set fixed train/val data split ratio')
    ap.add_argument('--seed', metavar='INT', type=int,
    default=None, help='Random seed for splitting data')
    ap.add_argument('--checkpoints', default=None, metavar='FILE',
    help='Save model checkpoints to directory')
    ap.add_argument('--save_model', default=None, metavar='FILE',
    help='Save model to file')
    ap.add_argument('--save_explanations', default=None, metavar='FILE',
    help='Save explanations to file')
    return ap



options = argparser().parse_args(sys.argv[1:])

print("Reading data", flush=True)

langs = ['en', 'fi', 'fr']

data_paths_v = {'en':options.data+'/en/all.tsv',
'fi':options.data+'/fi/all.tsv',
'fr':options.data+'/fr/all.tsv'
}

print("datapaths_v", data_paths_v)
dataset = train_multilabel.read_dataset(data_paths_v)

dataset = train_multilabel.resplit2(dataset)
print("Binarizing data", flush=True)
dataset = train_multilabel.binarize(dataset)
print("Ready to train:", flush=True)
model, tokenizer, report = train_multilabel.train(dataset, options)

report_file = options.save_explanations+'.eval.json'
print("Saving evaluation data to", report_file)
json.dump(report, open(report_file,'w'))

model.to('cuda')
print("Model loaded succesfully.")

print("Ready for explainability", flush=True)

for lang in langs:
    save_matrix = []
    save_attr = []
    print("Language:", lang)
    n_examples = len(dataset['validation_'+lang])
    print("XXX n examples in this languages", n_examples)
    for i in range(n_examples):
        if i % 100 == 0:
            print("Explaining example %d/%d" % (i, n_examples), flush=True)
        txt = dataset['validation_'+lang]['sentence'][i]
#        print("txt to be explained", txt)
        lbl = np.nonzero(dataset['validation_'+lang]['label'][i][0])[0]
        if txt == None:
            txt = " "   # for empty sentences
        target, aggregated, logits = explain_multilabel.explain(txt, model, tokenizer, int_bs=options.batch_size, n_steps=options.batch_size) #bs 44, nsteps 22?
 
        if target != None:
            # for all labels and their agg scores
            for tg, ag in zip(target[0], aggregated):
                target = tg
                aggregated = ag
                for j, (tok,a_val) in enumerate(aggregated[0]):
                    if j == 0:
                        #line = ['%s%d' % (['t','d'][val_indexes[i][0]], val_indexes[i][1]), str(lbl), target, logits]
                        line = ['%s%d' % ('d', i), str(lbl), target, logits]
                        save_matrix.append(line)
                    #line = ['%s%d' % (['t','d'][val_indexes[i][0]], val_indexes[i][1]), str(tok), a_val]
                    line = ['%s%d' % ('d', i), str(tok), a_val]
                    save_attr.append(line)
        else:  #for no classification, save none for target and a_val
            #line = ['%s%d' % (['t','d'][val_indexes[i][0]], val_indexes[i][1]), str(lbl), "None", logits]
            line = ['%s%d' % ('d', i), str(lbl), "None", logits]
            save_matrix.append(line)
            for word in txt.split():
                #line = ['%s%d' % (['t','d'][val_indexes[i][0]], val_indexes[i][1]), word, "None"]
                line = ['%s%d' % ('d', i), word, "None"]
                save_attr.append(line)
                #line = ['document_'+str(i), str(lbl), "None", word, "None", logits]
                #save_matrix.append(line)
        if i % 1000 == 999:
            pd.DataFrame(save_matrix).to_csv(options.save_explanations+'p_'+lang+'.tsv', sep="\t")
            pd.DataFrame(save_attr).to_csv(options.save_explanations+'a_'+lang+'.tsv', sep="\t")

    pd.DataFrame(save_matrix).to_csv(options.save_explanations+'p_'+lang+'.tsv', sep="\t")
    pd.DataFrame(save_attr).to_csv(options.save_explanations+'a_'+lang+'.tsv', sep="\t")

    print("Explanations succesfully saved")

# nice colours :)
#print_aggregated(target,aggregated, lbl)
