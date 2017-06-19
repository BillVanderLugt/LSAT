import pandas as pd
from pprint import pprint
import re
import numpy as np
import os
import io
import collections

from seq2seq.models import BasicSeq2Seq, seq2seq, AttentionSeq2Seq
import numpy as np
from keras.utils.test_utils import keras_test

from feature_engineering import save_pickle, load_pickle, \
                            save_spacydoc, save_spacydocs, load_spacey

def edit_text():
    '''
    Key in and edit three different versions for rule labels.
    '''
    print ("OK let's input some data...")
    for idx, row in labdf.iterrows():
        print ('ROW NO: ', idx)
        print (row['extracond'])
        print (row['text_list'])
        if row['net_label']=='':
            print ('no net label yet...')
        else:
            print ('existing label: ', row['net_label'])
            typed = input("EDIT? y/n?")
            if (typed == '') or (typed == 'n'):
                continue
        net_lab = []
        print ('inputting new net label...')
        for token in row['text_list']:
            print ('What about {} '.format(token))
            typed = input("'abcd...'=var ' 'keep ''stopword '#'=numeric 'q'quit ?")
            if typed == 'q':
                save_pickle(labdf, 'y_labels_new_df')
                print ('quitting')
                return labdf
            elif typed == '':
                continue
            elif typed == ' ':
                net_lab.append(token)
            elif typed == '0':
                net_lab.append('0')
            elif typed.isalpha():
                net_lab.append(typed.upper())
        print ('OK, new net lab is:', net_lab)
        typed = input('OK?')
        if typed == 'n':
            net_lab = [] #reset and redo
            for token in row['text_list']:
                print ('What about {} '.format(token))
                typed = input("'abcd...'=var ' 'keep ''stopword '#'=numeric 'q'quit ?")
                if typed == 'q':
                    save_pickle(labdf, 'y_labels_new_df')
                    print ('saving and quitting')
                    return labdf
                elif typed == '':
                    continue
                elif typed == ' ':
                    net_lab.append(token)
                elif typed == '0':
                    net_lab.append('0')
                elif typed.isalpha():
                    net_lab.append(typed.upper())
            print ('OK, new net lab is:', net_lab)
            typed = input('OK?')
            if typed == 'n':
                save_pickle(labdf, 'y_labels_new_df')
                print ('saving and quitting')
                return labdf
        else:
            labdf.set_value(idx, 'net_label', net_lab)
    save_pickle(labdf, 'y_labels_new_df')
    print ('done')
    return labdf

def create_net_labels():
    '''
    Key in and edit labels to use as targets for seq2seq.

    Return: a labels dataframe containing columns for:
        --game number
        --rule number
        --label (and's and or's have been expanded out)
        --condensed (and's and or's are left condensed, yet but's with implied subjects are expanded)
            * Note: Seq2seq was trained on this representation, so my expansion methods will still be needed
                for minor post-processing following seq2seq's predictions.
        --extra condensed (everything condensed)
    '''
    has_label = {key: val for key, val in label.items() if (val[0]!='?') and (val[0]!='q')}
    new = []
    for key, val in has_label.items():
        text = []
        for pair in val[1]:
            text.append(pair[0])
        cond = cond_label[key][0]
        xcond = xcond_label[key][0]
        new.append([key[0], key[1], val[0], cond, xcond, text])
    labdf = pd.DataFrame(new, columns = ['game_num', 'rule_num', 'label', 'condensed', 'extracond', 'text_list'])
    labdf['net_label'] = ''
    save_pickle(labdf, 'y_labels_new_df')

    return labdf

def create_lists():
    '''
    Collect lists of tokens to keep (keepers), stopwords, and mixed (in case some mistakenly labelled as both).

    Return: lists of keepers, stopwords, mixed
    '''
    keepers = set()
    stopwords = set()
    mixed = set()
    for idx, row in labdf.iterrows():
        orig_tokens = [token.lower() if len(token)>1 else token for token in row.text_list ]
        net_label = [token.lower() if len(token)>1 else token for token in row.net_label]
        keepers.update(net_label)
        stopwords.update(set(orig_tokens)-set(net_label))
        if len(row.net_label[0])>1: # uncapitalize first word of net_label if not a variable
            new = [row.net_label[0].lower()] + row.net_label[1:]
            print (new)
            labdf.set_value(idx, 'lowercase', new)
    mixed = (keepers & mixed)
    return keepers, stopwords, mixed

def get_Y_tokens():
    '''
    Collect set of tokens in seq2seq targets.

    Return: set of tokens
    '''

    tokens = set()
    for idx, row in labdf.iterrows():
        from_single_rule = set([token for token in row.tokenized])
        tokens.update(from_single_rule)
    return tokens

def swap_nums(old_label):
    '''
    Swap in generic number tag '0' for numbers that appear as tokens.

    Return: new version of label
    '''
    new_label = re.sub(r'[\d|*]', '0', old_label)
    return new_label

def tokenizer(old_label):
    '''
    Tokenize target labels, preserving blocks while separating all other characters.

    Return: list of tokens
    '''
    for block in blocks:
        head, sep, tail = old_label.partition(block)
        if head == old_label: # not find this label
            continue # keep looking
        return tokenizer(head) + [sep] + tokenizer(tail)
    return [char for char in old_label]

def get_lengths():
    '''
    Determine longest labels for seq2seq sources and targets.

    Return: length of longest source, length of longest target
    '''
    for idx, row in labdf.iterrows():
        labdf.set_value(idx, 'X_len', len(row.net_label))
        labdf.set_value(idx, 'Y_len', len(row.tokenized))
    return int(labdf.X_len.max()), int(labdf.Y_len.max())

def fill_seq2seq(src_col, vocab):
    '''
    Prepare strings and vocab dictionaries for seq2seq.

    Input: column to use for seq2seq source, vocabulary of source
    Return: text file in seq2seq format, dictionary of vocabulary indices
    '''
    idx_dict = {}
    output_str = ''
    for i, token in enumerate(vocab):
        idx_dict[token] = i
    for i, row in labdf.iterrows():
        line = []
        for j, token in enumerate(row[src_col]): # iterate through list
            line.append(str(idx_dict[token]))
        output_str += ' '.join(line) + '\n'
    return output_str, idx_dict

def gen_train_dev_test_idx(dev_samps=10, test_samps=5):
    '''
    Create indices for training, development, and test sets.
    Pull out 5 questions from game #145 and collect as test set.
    Randomly populate the dev set with 15 samples.
    Put remaining 122 samples into training set.

    Input: number of dev samples and number of test set samples.
    Return: indices for training, development, and test sets
    '''
    np.random.seed(seed=1)
    idx = list(range(0,30)) + list(range(35,labdf.shape[0]))
    np.random.shuffle(idx)
    test_idx = list(range(30,35)) # cherry pick questions for game 145
    dev_idx = idx[:dev_samps]
    train_idx = idx[dev_samps:]
    print ('len of train: ', len(train_idx),\
            'len of dev: ', len(dev_idx),\
            'len of test: ', len(test_idx))
    return train_idx, dev_idx, test_idx

def get_key_for_val(dict, val):
    #print ('val= ', val)
    return list(dict.keys())[list(dict.values()).index(val)]

def get_just_idx(idx):
    source_split = [all_source_split[i] for i in idx]
    target_split = [all_target_split[i] for i in idx]
    return '\n'.join(source_split), '\n'.join(target_split),\
            source_split, target_split

def print_source_target(source_split, target_split):
    for source, target in zip(source_split, target_split):
        print ('SOURCE: ', end='')
        for s in source.split(' '):
            s = int(s)
            print (get_key_for_val(X_dict, s), end=' ')
        print ('                    TARGET: ', end='')
        for t in target.split(' '):
            t = int(t)
            print (get_key_for_val(Y_dict, t), end='')
        print ()

def print_results(source_split, target_split):
    correct = 0
    for source, target in zip(source_split, target_split):
        source, target = source.strip(), target.strip()
        print ('PREDICTED: ', end='')
        for s in source.split(' '):
            s = int(s)
            print (get_key_for_val(Y_dict, s), end='')
        print ('        CORRECT: ', end='')
        for t in target.split(' '):
            t = int(t)
            print (get_key_for_val(Y_dict, t), end='')
        if source == target:
            print ("* CREDITED *")
            correct += 1
        print ()
    print ('ANSWERED {} of 15 CORRECTLY (ASSUMING IDENTICAL)'.format(correct))

def write_parallel_text(sources, targets, output_prefix):
  """
  Modification of Google's script to create source and target files.

  Writes two files where each line corresponds to one example
    - [output_prefix].sources.txt
    - [output_prefix].targets.txt

  Args:
    sources: Iterator of source strings
    targets: Iterator of target strings
    output_prefix: Prefix for the output file
  """
  source_filename = output_prefix + "sources.txt"
  target_filename = output_prefix + "targets.txt"

  with io.open(source_filename, "w", encoding='utf8') as source_file:
    for record in sources:
      source_file.write(record)
  print("Wrote {}".format(source_filename))

  with io.open(target_filename, "w", encoding='utf8') as target_file:
    for record in targets:
      target_file.write(record)
  print("Wrote {}".format(target_filename))

def generate_vocab(all_split):

    """
    MODIFICATION OF GOOGLE SCRIPT: https://github.com/google/seq2seq/blob/master/bin/tools/generate_vocab.py
    Generate vocabulary for a tokenized text file.
    """
    # Counter for all tokens in the vocabulary
    cnt = collections.Counter()
    output = ''

    for line in all_split:
        tokens = line.strip().split()
        cnt.update(tokens)

    print("Found {} unique tokens in the vocabulary.".format(len(cnt)))

    # Sort tokens by 1. frequency 2. lexically to break ties
    word_with_counts = cnt.most_common()
    word_with_counts = sorted(
        word_with_counts, key=lambda x: (x[1], x[0]), reverse=True)
    for word, count in word_with_counts:
        print("{}\t{}".format(word, count))
        output += "{}\t{}".format(word, count) + '\n'
    return output

def load(filename):
    '''
    Load a text file and return its contents.
    '''
    with open(filename) as f:
        data = f.readlines()
    return data

if __name__ == '__main__':
    '''
    Prepare data for subsequent training, via bash script, of seq2seq model.
    '''

    blocks = set([' and ', ' or ', ' ^ ', '==', 'not', 'abs', 'if ', ': '])
    label = load_pickle('rule_labels')
    cond_label = load_pickle('rule_cond_labels')
    xcond_label = load_pickle('rule_xcond_labels')

    pd.set_option("display.max_rows", 999)
    pd.set_option("display.max_columns", 20)
    pd.set_option('display.max_colwidth', 100)

    #labdf = create_net_labels()
    labdf = load_pickle('y_labels_new_df')

    labdf['lowercase'] = labdf['net_label'] # create row to hold lowercased version
    X_vocab, X_stopwords, X_mixed = create_lists() # Xs
    labdf['net_label'] = labdf['lowercase'] # swap lowercased version back in
    swapped = labdf.condensed.apply(swap_nums)
    labdf['swapped'] = swapped
    labdf.set_value(135, 'swapped', '((if (B<C): (A==0)) and (if (not(B<C)): (not(A==0))))') # correction
    tokenized = labdf.swapped.apply(tokenizer)
    labdf['tokenized'] = tokenized # Ys

    print (labdf[['swapped', 'tokenized']]) # Ys

    X_vocab_ct = len(X_vocab)
    X_vocab = list(X_vocab)
    X_vocab.sort()
    Y_vocab = list(get_Y_tokens())
    Y_vocab.sort()
    Y_vocab_ct = len(Y_vocab)
    print ('Vocab size of X: {}, size of Y: {}'.format(X_vocab_ct, Y_vocab_ct))
    X_longest, Y_longest = get_lengths()
    print ('Longest X: {}, Longest Y: {}'.format(X_longest, Y_longest))

    source_seq2seq, X_dict = fill_seq2seq('net_label', X_vocab)
    target_seq2seq, Y_dict = fill_seq2seq('tokenized', Y_vocab)

    all_source_split = source_seq2seq.split('\n')
    all_target_split = target_seq2seq.split('\n')

    train_idx, dev_idx, test_idx = gen_train_dev_test_idx()

    train_source, train_target, tr_src_split, tr_tgt_split = get_just_idx(train_idx)
    dev_source, dev_target, dv_src_split, dv_tgt_split = get_just_idx(dev_idx)
    test_source, test_target, ts_src_split, ts_tgt_split = get_just_idx(test_idx)

    print ('-------------TRAIN SOURCE AND TARGET------------------')
    print_source_target(tr_src_split, tr_tgt_split)
    print ('--------------DEV SOURCE AND TARGET------------------')
    print_source_target(dv_src_split, dv_tgt_split)
    print ('--------------TEST SOURCE AND TARGET------------------')
    print_source_target(ts_src_split, ts_tgt_split)

    write_parallel_text(train_source, train_target, '/Users/bvl/CAPSTONE_data/Capstone_seq2seq_data/train/')
    write_parallel_text(dev_source, dev_target, '/Users/bvl/CAPSTONE_data/Capstone_seq2seq_data/dev/')
    write_parallel_text(test_source, test_target, '/Users/bvl/CAPSTONE_data/Capstone_seq2seq_data/test/')
    tr_src_vocab = generate_vocab(tr_src_split)
    tr_tgt_vocab = generate_vocab(tr_tgt_split)
    write_parallel_text(tr_src_vocab, tr_tgt_vocab, '/Users/bvl/CAPSTONE_data/Capstone_seq2seq_data/train/vocab.')

    ###############################################################################################
    ### PRINT RESULTS AFTER USING BASH SCRIPTS TO TRAIN SEQ2SEQ MODEL AND PREDICT TARGET LABELS ###
    ###############################################################################################
    predict_split = load('/Users/bvl/CAPSTONE_data/Capstone_seq2seq_data/pred/dev_predictions.txt')
    correct_split = load('/Users/bvl/CAPSTONE_data/Capstone_seq2seq_data/dev/targets.txt')
    print ('DEV RESULTS:')
    print_results(predict_split, correct_split)

    predict_split = load('/Users/bvl/CAPSTONE_data/Capstone_seq2seq_data/pred/test_predictions.txt')
    correct_split = load('/Users/bvl/CAPSTONE_data/Capstone_seq2seq_data/test/targets.txt')
    print ('TEST RESULTS:')
    print_results(predict_split, correct_split)
