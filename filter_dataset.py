from datasets import load_dataset

# this was used to make English train data smaller

dataset = load_dataset(
        'csv', 
        data_files={
        'train': '../register_data/eacl/en/train.tsv', 
        'validation': '../register_data/eacl/en/dev.tsv', 
        'test': '../register_data/eacl/en/test.tsv'
        },
        delimiter='\t', 
        column_names=['label', 'sentence']
        )
# Take every 12th (8,3%)
dataset['train'] = dataset['train'].filter(lambda example, idx: idx % 12 == 0, with_indices=True)

dataset['train'].to_csv('../register_data/eacl/en/train_filt.tsv', sep='\t', index = False, columns = None)
