from captum.attr import visualization as viz
from captum.attr import IntegratedGradients, LayerConductance, LayerIntegratedGradients
from captum.attr import configure_interpretable_embedding_layer, remove_interpretable_embedding_layer
import torch
import transformers
from transformers import AutoTokenizer
import captum
import re
from datasets import load_dataset
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
from datasets import load_dataset
import pandas as pd
import csv
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import sys


labels = ['HI', 'ID', 'IN', 'IP', 'LY', 'NA', 'OP', 'SP', 'av', 'ds', 'dtp', 'ed', 'en', 'fi', 'it', 'lt', 'mt', 'nb', 'ne', 'ob', 'ra', 're', 'rs', 'rv', 'sr']
file_name = ".tsv"
int_bs = 10
data_name = ""   #path and .pkl already in code
model_name = ""  #path and .pt already in code


def argparser():
    ap = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    ap.add_argument('--model_name', default=model_name,
                    help='Pretrained model name')
    ap.add_argument('--data', metavar='FILE', required=True,
                    help='Path to test datasets')
    ap.add_argument('--int_batch_size', metavar='INT', type=int,
                    default=int_bs,
                    help='Batch size for integrated gradients')
    ap.add_argument('--seed', metavar='INT', type=int,
                    default=None, help='Random seed for splitting data')
    ap.add_argument('--file_name', default=None, metavar='FILE',
                    help='Path to file and the beginning of the filename')
    #ap.add_argument('--save_predictions', default=False, action='store_true',
    #                help='save predictions and labels for dev set, or for test set if provided')
    return ap




# # Forward on the model -> data in, prediction out, nothing fancy really
def predict(model, inputs, attention_mask=None):
    pred=model(inputs, attention_mask=attention_mask) # TODO: batch_size?
    return pred.logits #return the output of the classification layer

def blank_reference_input(tokenized_input, blank_token_id): #b_encoding is the output of HFace tokenizer
    """
    makes a tuple of blank (input_ids, token_type_ids, attention_mask)
    right now position_ids, and attention_mask simply point to tokenized_input
    """

    blank_input_ids=tokenized_input.input_ids.clone().detach()
    blank_input_ids[tokenized_input.special_tokens_mask==0]=blank_token_id #blank out everything which is not special token
    return blank_input_ids, tokenized_input.attention_mask

def summarize_attributions(attributions):
    attributions = attributions.sum(dim=-1).squeeze(0)
    attributions = attributions / torch.norm(attributions)
    return attributions

def aggregate(inp,attrs,tokenizer):
    """detokenize and merge attributions"""
    detokenized=[]
    for l in inp.input_ids.cpu().tolist():
        detokenized.append(tokenizer.convert_ids_to_tokens(l))
    attrs=attrs.cpu().tolist()

    aggregated=[]
    for token_list,attr_list in zip(detokenized,attrs): #One text from the batch at a time!
        res=[]
        #print(token_list)
        for token,a_val in zip(token_list,attr_list):
            if token == "<s>" or token == "</s>":  # special tokens
                res.append((token,a_val))
            elif token.startswith("▁"):
                #This NOT is a continuation. A NEW word.
                res.append((token[1:],a_val))
                #print(res)
            else:  # we're continuing a word and need to choose the larger abs value of the two
                last_a_val = res[-1][1]
                #print("last val", last_a_val)
                if abs(a_val)<abs(last_a_val): #past value bigger
                    res[-1]=(res[-1][0]+token, last_a_val)
                else:  #new value bigger
                    res[-1]=(res[-1][0]+token, a_val)

        aggregated.append(res)
    return aggregated


def explain(text,model,tokenizer,wrt_class="winner", int_bs=10, n_steps=50):

    # Tokenize and make the blank reference input
    inp = tokenizer(text,return_tensors="pt",return_special_tokens_mask=True,truncation=True).to(model.device)
    b_input_ids, b_attention_mask=blank_reference_input(inp, tokenizer.convert_tokens_to_ids("-"))


    def predict_f(inputs, attention_mask=None):
        return predict(model,inputs,attention_mask)

    lig = LayerIntegratedGradients(predict_f, model.roberta.embeddings)
    if wrt_class=="winner":
        # make a prediction
        prediction=predict(model,inp.input_ids, inp.attention_mask)
        # get the classification layer outputs
        logits = prediction.cpu().detach().numpy()[0]
        # calculate sigmoid for each
        sigm = 1.0/(1.0 + np.exp(- logits))
        # make the classification, threshold = 0.5
        target = np.array([pl > 0.5 for pl in sigm]).astype(int)
        # get the classifications' indices
        target = np.where(target == 1)
        # return None if no classification was done
        if len(target[0]) == 0:
            return None, None, logits

    else:
        # not implemented really
        target = wrt_class


    aggregated = []
    # loop over the targets
    for tg in target[0]:
        attrs, delta= lig.attribute(inputs=(inp.input_ids,inp.attention_mask),
                                     baselines=(b_input_ids,b_attention_mask),
                                     return_convergence_delta=True,target=tuple([np.array(tg)]),internal_batch_size=int_bs,n_steps=n_steps)
        # append the calculated and normalized scores to aggregated
        attrs_sum = attrs.sum(dim=-1)
        attrs_sum = attrs_sum/torch.norm(attrs_sum)
        aggregated_tg=aggregate(inp,attrs_sum,tokenizer)
        aggregated.append(aggregated_tg)

    # these are wonky but will have dim numberofpredictions x 1
    return target,aggregated,logits


def print_aggregated(target,aggregated,real_label):
    """"
    This requires one target and one agg vector at a time
    Shows agg scores as colors
    """
    print("<html><body>")
    for tg,inp_txt in zip(target,aggregated): #one input of the batch
        x=captum.attr.visualization.format_word_importances([t for t,a in inp_txt],[a for t,a in inp_txt])
        print(f"<b>prediction: {labels[tg[0]]}, real label: {real_label}</b>")
        print(f"""<table style="border:solid;">{x}</table>""")
    print("</body></html>")

def print_scores(target, aggregated, idx):
    """"
    Prints doc_id, label, token and agg score
    Mainly used for testing
    """
    for tg, ag in zip(target[0], aggregated):
        target = tg
        aggregated = ag
        for tok,a_val in aggregated[0]:
            if a_val > 0.05:
                #print(f"{counter}",item['label'],label_enc_rev[target.item()],tok,a_val,sep="\t")
                print("document_"+str(idx),labels[target],str(tok),a_val,sep="\t")

def remove_NA(d):
  """ Remove null values and separate multilabel values with comma """
  if d['label'] == None:
    d['label'] = np.array('NA')
  if ' ' in d['label']:
    d['label'] = ",".join(sorted(d['label'].split()))
  return d


def label_encoding(d):
  """ Split the multi-labels """
  d['label'] = np.array(d['label'].split(","))
  return d

def binarize(dataset):
    """ Binarize the labels of the data. Fitting based on the whole data. """
    mlb = MultiLabelBinarizer()
    mlb.fit([labels])
    print("Binarizing the labels:")
    dataset = dataset.map(lambda line: {'label': mlb.transform([line['label']])})
    return dataset


if __name__=="__main__":
    options = argparser().parse_args(sys.argv[1:])
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    model = torch.load(options.model_name)
    model.to('cuda')
    print("Model loaded succesfully.")

    # load the test data
    path = options.data
    dataset = load_dataset(
              'csv',
              data_files = {
                  'en': path+'en/test.tsv-simp.tsv',
                  'fi': path+'fi/test.tsv-simp.tsv',
                  'fr': path+'fr/test.tsv-simp.tsv',
                  'sv': path+'sv/test.tsv-simp.tsv',
              }, delimiter='\t',
              column_names=['label', 'sentence']
              )
    print("Dataset loaded succesfully.")


    dataset = dataset.map(remove_NA)
    dataset = dataset.map(label_encoding)
    dataset = binarize(dataset)


    print("Ready for explainability")

    # loop over languages
    for key in {'en', 'fi', 'fr', 'sv'}:

        save_matrix = []

        for i in range(len(dataset[key])):
          #print(i)
          txt = dataset[key]['sentence'][i]
          lbl = np.nonzero(dataset[key]['label'][i][0])[0]
          if txt == None:
             txt = " "   # for empty sentences
          target, aggregated, logits = explain(txt, model, tokenizer, int_bs=options.int_batch_size)
          if target != None:
             # for all labels and their agg scores
             for tg, ag in zip(target[0], aggregated):
               target = tg
               aggregated = ag
               for tok,a_val in aggregated[0]:
                    line = ['document_'+str(i), str(lbl), target, str(tok), a_val, logits]
                    save_matrix.append(line)
          else:  #for no classification, save none for target and a_val
             for word in txt.split():
               line = ['document_'+str(i), str(lbl), "None", word, "None", logits]
               save_matrix.append(line)

        filename = options.file_name+key+'.tsv'
        pd.DataFrame(save_matrix).to_csv(filename, sep="\t")
        print("Dataset "+ key +" succesfully saved")

    # nice colours :)
    #print_aggregated(target,aggregated, lbl)
