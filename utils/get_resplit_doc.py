import train_multilabel
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter



def argparser():
    ap = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    ap.add_argument('--data', metavar='FILE', required=True,
    help='Path to binarized dataset')
    ap.add_argument('--split', metavar='FLOAT', type=float,
    default=None, help='Set fixed train/val data split ratio')
    ap.add_argument('--seed', metavar='INT', type=int,
    default=None, help='Random seed for splitting data')
    return ap

options = argparser().parse_args(sys.argv[1:])


print("Reading data", flush=True)
dataset = train_multilabel.read_dataset(options.data)
print("Splitting data", flush=True)
dataset = train_multilabel.resplit(dataset, ratio=options.split, seed=options.seed)

for i in range(N_DOCS):
    # Get internal doc indices from experiments output file
    dataset['validation'][i]['sentence']
