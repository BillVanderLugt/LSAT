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

def gen_spacy_docs(df, source, dest, dest_agg):
    docs = []
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
        docs.append(merged) # add the merged sentences from one game to the whole set
    return docs

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

def gen_tfidf(texts):
    vectorizer = TfidfVectorizer()
    data = vectorizer.fit_transform(texts)
    return data

pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', SGDClassifier()),
])

parameters = {
    'vect__stop_words': ('english', None),
    #'vect__max_df': (0.3, 0.5, 0.7, 1.0),
    'vect__min_df': (0.0, 0.05, .1),
    #'vect__max_features': (None, 50, 1000),
    #'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
    'tfidf__use_idf': (True, False),
    #'tfidf__norm': ('l1', 'l2'),
    'clf__alpha': (1.0, 0.5, 0.1, 0.01),
    #'clf__penalty': ('l2'),
    'clf__n_iter': (5, 10, 50, 100, 500),
    'clf__loss': ('huber', 'log')
}

# parameters = {
#     'vect__max_df': (0.5, 1.0),
#     'vect__min_df': (0.0, 0.1),
#     #'vect__max_features': (None, 50, 1000),
#     #'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
#     'tfidf__use_idf': (True, False),
#     #'tfidf__norm': ('l1', 'l2'),
#     'clf__C': (0.1, 1.0, 2.0, 4.0),
#     'clf__kernel': ('linear', 'poly', 'rbf'),
#     #'clf__penalty': ('l2'),
#     #'clf__n_iter': (5, 10, 50)
#     #'clf__loss': ('huber', 'log', 'modified_huber', 'squared_hinge')
# }
#(C=1.0, kernel='rbf', degree=3, gamma='auto', coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape=None, random_state=None)[source]

def grid(X, y):
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1, cv=8)

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

def CV_eval(model, X, y):
    scores = cross_val_score(model, X, y, cv=5)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    return scores.mean()

def prep_X_y_classifier(df, source, y):
    docs = []
    Y = []
    counter = 1
    for game in df.iterrows():
        #print ('############## collecting texts for game #: {} #################'.format(counter))
        joined = ' '.join(source[game[0]])
        #print ('JOINED:')
        #print (joined)
        docs.append(joined)
        Y.append(y[game[0]])
        counter += 1
    idx = df.index
    #print ('len of docs {}, Y {}, idx {}'.format(len(docs), len(Y), len(idx)))
    return docs, Y, idx

def balance_classes(df):
    y = np.zeros(Lsat.df.shape[0])
    bl = df[df.primary_type == 'Basic Linear'].index
    gr = df[df.primary_type == 'Grouping'].index
    ps = df[df.primary_type == 'Pure Sequencing'].index
    y[ps] = 0
    y[bl] = 1
    y[gr] = 2

    bl_samp = np.random.choice(bl, size=16, replace=False)
    gr_samp = np.random.choice(gr, size=16, replace=False)

    # print (bl_samp, len (bl_samp))
    # print (gr_samp, len (gr_samp))
    # print (ps, len (ps))
    index = np.concatenate((gr_samp, bl_samp, ps))
    # print (index)
    X = df.loc[index]
    y = y[index]
    #print (y, len(y))
    return X, y

'''
Int64Index([  3,   5,   6,   9,  10,  12,  13,  14,  17,  18,  19,  22,  25,
             28,  29,  33,  35,  38,  41,  45,  47,  50,  51,  53,  54,  57,
             61,  64,  65,  69,  70,  77,  79,  80, 145, 165, 186, 206, 216,
            220, 228, 230, 239, 262, 265, 268, 270, 271, 272, 273, 277, 280,
            282, 286, 293, 297, 301, 304],
'''

if __name__ == '__main__':
    Lsat = load_pickle('LSAT_data')
    print ('loading spacy...')
    nlp = spacy.load('en')

    extract_parentheticals(Lsat.keyed_seq, Lsat.prompts, Lsat.prompts_as_list)

    prompt_docs = gen_spacy_docs(Lsat.keyed, Lsat.prompts, Lsat.prompts_as_spdoc, Lsat.prompts_1_spdoc)
    rules_docs = gen_spacy_docs(Lsat.keyed, Lsat.rules, Lsat.rules_as_spdoc, Lsat.rules_1_spdoc)

    # all_keyed_prompts_tfidf = gen_tfidf()
    # all_keyed_rules_tfidf = gen_tfidf(rules_docs)

    # In [44]: all_keyed_prompts_tfidf.shape
    # Out[44]: (164, 781)
    #
    # In [45]: all_keyed_rules_tfidf.shape
    # Out[45]: (265, 510)

    balanced, y = balance_classes(Lsat.keyed)
    X_prompts, _ , idx = prep_X_y_classifier(balanced, Lsat.prompts, Lsat.df.primary_type)
    X_rules, _ , idx = prep_X_y_classifier(balanced, Lsat.rules, Lsat.df.primary_type)

    # Tfidf_prompts = gen_tfidf(X_prompts).todense()
    # Tfidf_rules = gen_tfidf(X_rules).todense()
    # print (type(Tfidf_prompts), type(Tfidf_rules), Tfidf_prompts.shape, Tfidf_rules.shape)
    # Tfidf_combined = np.concatenate((Tfidf_prompts, Tfidf_rules), axis=1)


    print ("starting grid search using prompts...")
    grid(X_prompts, y)
