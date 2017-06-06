import spacy
import pickle
from load_categories_df import LSAT
import re
from string import punctuation

from pprint import pprint
from time import time
import logging
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline


def save_pickle(file, name):
    with open("../data/" + name + ".pkl", 'wb') as f:
        pickle.dump(file, f)
    print ("done pickling ", name)

def load_pickle(name):
    '''
    returns unpickled file
    '''
    with open("../data/" + name + ".pkl", 'rb') as f_un:
        file_unpickled = pickle.load(f_un)
    print ("done unpickling ", name)
    return file_unpickled

def strip_sent(sent):
    pattern = re.compile('\w+')
    return pattern.findall(sent)

def clean_sent(sent):
    return " ".join([clean_word(w) for w in sent.split(' ')])

def clean_word(old):
    '''
    eliminates parentheses and dashes
    '''
    word = old
    if word == '--':
        word = ''
    word = re.sub(r'[().,:]', '', word) # strip certain punctuation
    word = re.sub(r"'s", '', word) # strip possessives
    # if word != old:
    #     print ("@@@@@@ swapping in {} for {} @@@@@".format(word, old))
    return word

def sent_pos(sent):
    cleaned = clean_sent(sent)
    cleaned_as_list = cleaned.split(' ')
    doc = nlp(cleaned)
    out_plus_punct = []
    out = []
    for word in doc:
        if word.tag_ !='POS': # ignore possessives
            out_plus_punct.append(word.tag_) # collect version including punctuation
            #print ('testing ', word.tag_[0], " isalnum=", word.tag_[0].isalnum())
            if word.tag_[0].isalnum(): # winnow out punctuation
                out.append(word.tag_)
    # print ("Pos without punct:", out)
    # print ("Pos with punct:", out_plus_punct)
    if len(cleaned_as_list) != len(out):
        print ("$$$$$$$$$$$$$$$$$$$$$ ERROR $$$$$$$$$$$$$$$$$$$$$$$")
        print ("Len of original: {}   Len of parsed sans punct: {}".\
                        format(len(cleaned_as_list), len(out)))
        for i, w in enumerate(cleaned_as_list):
            print (w, out[i])
    return cleaned_as_list, out, out_plus_punct

def _tag_sent(sent):
    # for word in sent.split(' '):
    #     print (word.ljust(12), end='')
    # print (end='\n')
    as_list, parts_of_speech, pos_plus_punct = sent_pos(sent)
    # for word in parts_of_speech:
    #     print (word.ljust(12), end='')
    # print (end='\n\n')
    return as_list, parts_of_speech, pos_plus_punct

def tag(df, source, destination_list, pos_dest, dest_plus_punct):
    for game in df.iterrows():
        #print ('############## tagging game #: {} #################'.format(game[0]))
        as_list = []
        pos_list = []
        pos_plus_punct = []
        for sent in source[game[0]]:
            sent_list, pos, plus_punct = _tag_sent(sent)
            as_list.append(sent_list)
            pos_list.append(pos)
            pos_plus_punct.append(plus_punct)

        destination_list[game[0]] = as_list
        pos_dest[game[0]] = pos_list
        dest_plus_punct[game[0]] = pos_plus_punct
    return pos_list, pos_plus_punct

def gen_spacy_docs(df, source, dest, dest_agg):
    for game in df.iterrows():
        # print ('############## extracting SpaCy docs from game #: {} #################'.format(game[0]))
        dest_list = []
        merged = []
        for sent in source[game[0]]:
            doc = nlp(sent) # generate SpaCy doc object
            dest_list.append(doc) # store in destination Lsat attribute
            merged += doc
        dest[game[0]] = dest_list
        dest_agg[game[0]] = merged
    return

def _extract_paren(sent):
    regex = re.compile(r'\([^)]*\) ?')
    paren = regex.findall(sent)
    #print (paren)
    new_sent = regex.sub('', sent)
    #print (new_sent)
    return new_sent, paren

def extract_parentheticals(df, source, destination_list):
    for game in df.iterrows():
        # print ('############## extracting parentheticals from game #: {} #################'.format(game[0]))
        stripped_sents = []
        paren_list = []
        for sent in source[game[0]]:
            stripped_sent, paren = _extract_paren(sent)
            stripped_sents.append(stripped_sent)
            extracted_words = [x[1:-1].split(' ') for x in paren]
            paren_list.append(extracted_words)
        destination_list[game[0]] = stripped_sents
        Lsat.parentheticals[game[0]] =  extracted_words # remove parens & separate
    return stripped_sents, paren_list

def scan_for_term(df, source, term):
    for game in df.iterrows():
        #print ('############## processing game #: {} #################'.format(game[0]))
        for sent in source[game[0]]:
            if term in sent:
                print ("Found {} in game {}".format(term, game[0]))

def check_tagging(row_num=3):
    print ()
    print ('Checking pos tagging...')
    print ()
    for i, prompt in enumerate(Lsat.prompts[row_num]):
        print (prompt)
        print ('prompts_as_list:', Lsat.prompts_as_list[row_num][i])
        print ('prompts_pos_as_list:', Lsat.prompts_pos_as_list[row_num][i])
        print ()

    print ('parentheticals:', Lsat.parentheticals[row_num])

    for i, rule in enumerate(Lsat.rules[row_num]):
        print (rule)
        print (Lsat.rules_as_list[row_num][i])
        print (Lsat.rules_pos_as_list[row_num][i])
        print ()

def check_SpaCy(row_num=3):
    print ()
    print ('Checking SpaCy doc tagging...')
    print ()
    for i, prompt in enumerate(Lsat.prompts_as_spdoc[row_num]):
        print ("########### PROMPTS ###########")
        print (prompt)
        for w in Lsat.prompts_as_spdoc[row_num][i]:
            print (w.text, w.tag_)
        print ("AS SINGLE DOC:")
        print (Lsat.prompts_1_spdoc[row_num])

    for i, rule in enumerate(Lsat.rules[row_num]):
        print ("########### RULES ###########")
        for w in Lsat.rules_as_spdoc[row_num][i]:
            print (w.text, w.tag_)
        print ("AS SINGLE DOC:")
        print (Lsat.rules_1_spdoc[row_num])

def gen_tfidf(df, source, dest):
    docs = []
    for game in df.iterrows(): # collect docs from individual games' source
        #print ('############## processing game #: {} #################'.format(game[0]))
        docs += source[game[0]]
    vectorizer = TfidfVectorizer()
    data = vectorizer.fit_transform(docs)
    #data = data.toarray()
    for game, d in zip(df.iterrows(), data):
        #print ('############## processing game #: {} #################'.format(game[0]))
        dest[game[0]] = d
    return data

pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', SGDClassifier()),
])

parameters = {
    'vect__stop_words': ('english', None),
    'vect__max_df': (0.3, 0.5),
    #'vect__min_df': (.01, .1),
    #'vect__max_features': (None, 50, 1000),
    #'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
    #'tfidf__use_idf': (True, False),
    #'tfidf__norm': ('l1', 'l2'),
    'clf__alpha': (0.001, .0001),
    #'clf__penalty': ('l2'),
    'clf__n_iter': (5, 10),
    #'clf__loss': ('huber', 'log', 'modified_huber', 'squared_hinge')
}

# parameters = {
#     #'vect__max_df': (0.1, 0.3),
#
#     #'vect__max_features': (None, 50, 1000),
#     #'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
#     #'tfidf__use_idf': (True, False),
#     #'tfidf__norm': ('l1', 'l2'),
#     'clf__C': (0.001, 0.1, 1.0, 2.0),
#     'clf__kernel': ('linear', 'poly', 'rbf')
#     #'clf__penalty': ('l2'),
#     #'clf__n_iter': (30, 50),
#     #'clf__loss': ('huber', 'log', 'modified_huber', 'squared_hinge')
# }
#(C=1.0, kernel='rbf', degree=3, gamma='auto', coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape=None, random_state=None)[source]

def grid(X, y):
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1)

    print("Performing grid search...")
    print("pipeline:", [name for name, _ in pipeline.steps])
    print("parameters:")
    pprint(parameters)
    t0 = time()
    grid_search.fit(X, y)
    print("done in %0.3fs" % (time() - t0))
    print()

    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))

def eval(model, X, y):
    scores = cross_val_score(model, X, y, cv=5)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    return scores.mean()

def prep_X_y_classifier(Lsat.keyed):
    pass

if __name__ == '__main__':
    Lsat = load_pickle('LSAT_data')
    print ('loading spacy...')
    nlp = spacy.load('en')

    extract_parentheticals(Lsat.keyed_seq, Lsat.prompts, Lsat.prompts_as_list)

    # customized tokenization (works for all seq games) and tagging
    #tag(Lsat.keyed, Lsat.prompts, Lsat.prompts_as_list, Lsat.prompts_pos_as_list, Lsat.prompts_pos_plus_punct)
    #tag(Lsat.keyed, Lsat.rules, Lsat.rules_as_list, Lsat.rules_pos_as_list, Lsat.rules_pos_plus_punct)

    gen_spacy_docs(Lsat.keyed, Lsat.prompts, Lsat.prompts_as_spdoc, Lsat.prompts_1_spdoc)
    gen_spacy_docs(Lsat.keyed, Lsat.rules, Lsat.rules_as_spdoc, Lsat.rules_1_spdoc)

    all_keyed_prompts_tfidf = gen_tfidf(Lsat.keyed, Lsat.prompts, Lsat.prompts_tfidf)
    all_keyed_rules_tfidf = gen_tfidf(Lsat.keyed, Lsat.rules, Lsat.rules_tfidf)

    # In [44]: all_keyed_prompts_tfidf.shape
    # Out[44]: (164, 781)
    #
    # In [45]: all_keyed_rules_tfidf.shape
    # Out[45]: (265, 510)

    X, y = prep_X_y_classifier(Lsat.keyed)
    grid(X, y)
