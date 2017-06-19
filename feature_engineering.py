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
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

def save_pickle(file, name):
    '''
    Save pickled version of data at file path as name.
    '''
    with open("../data/" + name + ".pkl", 'wb') as f:
        pickle.dump(file, f)

def load_pickle(name):
    '''
    Return unpickled data from file name.
    '''
    with open("../data/" + name + ".pkl", 'rb') as f_un:
        file_unpickled = pickle.load(f_un)
    return file_unpickled

def save_spacydoc(sp_doc, name):
    '''
    Split a SpaCy doc object into separate files.
    Pickle those files.
    '''
    text = []
    pos = []
    pairs = []
    for game in sp_doc:
        game_text = []
        game_pos = []
        game_pairs = []
        for w in game:
            game_text.append(w.text)
            game_pos.append(w.tag_)
            game_pairs.append((w.text, w.tag_))
        text.append(game_text)
        pos.append(game_pos)
        pairs.append(game_pairs)
    save_pickle(text, name + '_text')
    save_pickle(pos, name + '_tags')
    save_pickle(pairs, name + '_pairs')

def save_spacydocs(sp_doc_list, name):
    '''
    Split nested SpaCy doc objects into separate files.
    Pickle those files.
    '''
    text = []
    pos = []
    pairs = []
    for game in sp_doc_list:
        game_text = []
        game_pos = []
        game_pairs = []
        for sent in game:
            sent_text = []
            sent_pos = []
            sent_pairs = []
            for w in sent:
                sent_text.append(w.text)
                sent_pos.append(w.tag_)
                sent_pairs.append((w.text, w.tag_))
            game_text.append(sent_text)
            game_pos.append(sent_pos)
            game_pairs.append(sent_pairs)
        text.append(game_text)
        pos.append(game_pos)
        pairs.append(game_pairs)
    save_pickle(text, name + '_text')
    save_pickle(pos, name + '_tags')
    save_pickle(pairs, name + '_pairs')

def save_spacydocs_dict(sp_doc_dict, name):
    '''
    Separate and save a dictionary containing SpaCy doc objects.
    '''
    text = {}
    pos = {}
    pairs = {}
    for key, val in sp_doc_dict.items():
        try:
            text[key] = val.text
            pos[key] = val.tag_
            pairs[key] = (val.text, val.tag_)
        except:
            print ("error saving game {}, question {} ".format(key[0], key[1]))

    save_pickle(text, name + '_text')
    save_pickle(pos, name + '_tags')
    save_pickle(pairs, name + '_pairs')

def load_spacey(name):
    '''
    Load lists containing tokens for the text, part-of-speech tags, and (text, pos) pairs.
    '''
    text = load_pickle(name + '_text')
    tags = load_pickle(name + '_tags')
    pairs = load_pickle(name + '_pairs')
    return text, tags, pairs

def gen_spacy_docs(df, source, dest, dest_agg):
    '''
    Generate SpaCy doc objects.

    Input: --df: DataFrame
           --source: source from which to draw
           --dest: destination
           --dest_agg: destination aggregated
    Return: list of SpaCy doc objects
    '''
    docs = []
    for game in df.iterrows():
        # print ('############## extracting SpaCy docs from game #: {} #################'.format(game[0]))
        dest_list = []
        merged = []
        for sent in source[game[0]]:
            doc = nlp(sent) # generate SpaCy doc object
            doc = [w for w in doc if w.tag_ != 'POS'] # eliminate possessives
            dest_list.append(doc) # store in destination Lsat attribute
            merged += doc
        dest[game[0]] = dest_list
        dest_agg[game[0]] = merged
        docs.append(merged) # add the merged sentences from one game to the whole set
    return docs

def gen_spacy_docs_dict(df, source, dest):
    '''
    Generate SpaCy doc object dictionaries.

    Input: --df: DataFrame
           --source: source from which to draw
           --dest: destination
    Return: list of SpaCy doc objects
    '''

    for game in df.iterrows():
        print ('############## extracting SpaCy docs from game #: {} #################'.format(game[0]))
        try: # skip if unkeyed
            #print (source[game[0]])
            for quest_num, answers in enumerate(source[game[0]]):
                #print ('question num ', quest_num)
                dest_list = []
                if answers != []:
                    for a, sent in enumerate(answers):
                        #print ('sent', sent)
                        doc = nlp(sent) # generate SpaCy doc object
                        doc = [w for w in doc if w.tag_ != 'POS'] # eliminate possessives
                        #print (doc)
                        dest_list.append(doc) # store doc objs in destination Lsat attribute
                        #print ('game[0], quest_num, ans_num', game[0], quest_num, a)
                    dest[(game[0], quest_num, a)] = dest_list
        except:
            print ('error in game ', source[game[0]])

def input_labels(row_num=3):
    '''
    Input labels.
    '''
    print ()
    for i, rule in enumerate(Lsat.rules[row_num]):
        pairs = []
        print()
        for w in Lsat.rules_as_spdoc[row_num][i]:
            print ((row_num, i), w.text, w.tag_)
            pairs.append((w.text, w.tag_))
        print (pairs)
        if (row_num, i) in label: # check for expanded label
            print ("Already labeled expanded: ", label[(row_num, i)][0])
            typed = input("Wanna edit the expanded(y)?  quit(q)? no(return)?")
            if typed == 'y':
                typed = input("What should the expanded be? ")
                pairs =label[(row_num, i)][1]
                label[(row_num, i)] = (typed, pairs)
                print ("new expanded label: ", label[(row_num, i)])
            elif typed == 'q':
                return False
        else:
            print ("No expanded label so...")
            typed = input("What expanded label for this new rule? ('' or space to exit) ")
            if typed == '' or typed == ' ':
                return False
            label[(row_num, i)] = (typed, pairs)
        if (row_num, i) in cond_label: # check for condensed label
            print ("Already labeled condensed: ", cond_label[(row_num, i)][0])
            typed = input("Wanna edit condensed? ('' no) (c/y yes)  (q quit)? ")
            if (typed == 'c') or (typed == 'y'):
                typed = input("What should the condensed be? ('' same) ")
                if typed == '':
                    cond_label[(row_num, i)] = label[(row_num, i)]
                else:
                    pairs = cond_label[(row_num, i)][1]
                    cond_label[(row_num, i)] = (typed, pairs)
                print ("new condensed label: ", cond_label[(row_num, i)])
            elif typed == 'q':
                return False
        else:
            print ("No condensed label so...")
            typed = input("What condensed label? ('' same as expanded)('q' quit) ")
            if typed == 'q':
                return False
            elif typed == '':
                cond_label[(row_num, i)] = label[(row_num, i)]
                print ("OK, both expanded and condensed are ", cond_label[(row_num, i)])
            else:
                cond_label[(row_num, i)] = (typed, pairs)
                print ("new condensed label: ", cond_label[(row_num, i)])
                print ()
        if (row_num, i) in xcond_label: # check for extra condensed label (implied subjects)
            print ("Already labeled extra condensed: ", xcond_label[(row_num, i)][0])
            typed = input("Wanna edit extra condensed? ('' no) (x/y yes)  (q quit)? ")
            if (typed == 'x') or (typed == 'y'):
                typed = input("What should the extra condensed be? ")
                if typed == '':
                    xcond_label[(row_num, i)] = cond_label[(row_num, i)]
                else:
                    pairs = xcond_label[(row_num, i)][1]
                    xcond_label[(row_num, i)] = (typed, pairs)
                print ("new extra condensed label: ", xcond_label[(row_num, i)])
            elif typed == 'q':
                return False
        else:
            print ("No extra condensed label so...")
            typed = input("What extra condensed label? ('' same as condensed)('q' quit) ")
            if typed == 'q':
                return False
            elif typed == '':
                xcond_label[(row_num, i)] = cond_label[(row_num, i)]
                print ("OK, both extra condensed and condensed are ", xcond_label[(row_num, i)])
            else:
                xcond_label[(row_num, i)] = (typed, pairs)
                print ("new extra condensed label: ", xcond_label[(row_num, i)])
                print ()
    return True

def check_all_SpaCy(df):
    '''
    Quality control SpaCy docs.
    '''
    for game in df.iterrows():
        print ('-----------------------------{}-----------------------------------'.format(game[0]))
        flag = input_labels(row_num=game[0])
        if flag == False:
            break
    save_pickle(label, 'rule_labels')
    save_pickle(cond_label, 'rule_cond_labels')
    save_pickle(xcond_label, 'rule_xcond_labels')
    return

# prep some variables for grid searches to tune text classification model

pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', SGDClassifier()),
])

parameters = {
    'vect__stop_words': (None, None),
    'vect__max_df': (0.3, 0.5, 0.7, 1.0),
    'vect__min_df': (0.0, 0.05, .1),
    #'vect__max_features': (None, 50, 100, 500),
    #'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
    'tfidf__use_idf': (True, False),
    #'tfidf__norm': ('l1', 'l2'),
    'clf__alpha': (2.0, 1.0, 0.5),
    #'clf__penalty': ('l2'),
    'clf__n_iter': (1, 3, 5),
    'clf__loss': ('huber', 'log')
}

def grid(X, y):
    '''
    Adapted from: http://scikit-learn.org/stable/auto_examples/model_selection/grid_search_text_feature_extraction.html#sphx-glr-auto-examples-model-selection-grid-search-text-feature-extraction-py
    Perform a grid search.
    '''

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
    '''
    Perform 8-fold cross-validation

    Input: model, X data, Y data
    Return: mean of cross-val accuracy scores
    '''
    scores = cross_val_score(model, X, y, cv=8)
    pprint (scores)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    return scores.mean()

def prep_X_y_classifier(df, source, y):
    '''
    Prepare data for text classification model.

    Input: DataFrame, source, y
    Return: docs, Y, indices
    '''
    docs = []
    Y = []
    counter = 1
    for game in df.iterrows():
        #print ('############## collecting texts for game #: {} #################'.format(counter))
        joined = ' '.join(source[game[0]])
        docs.append(joined)
        Y.append(y[game[0]])
        counter += 1
    idx = df.index
    return docs, Y, idx

def balance_classes(df):
    '''
    Ensure that classes comprising training data are balanced.
    Can be adapted for either 2 or 3 classes, as desired.

    Input: DataFrame
    Return: separate DataFrames for X and Y data
    '''

    bl = df[df.primary_type == 'Basic Linear'].index
    gr = df[df.primary_type == 'Grouping'].index
    ps = df[df.primary_type == 'Pure Sequencing'].index

    y = np.zeros(Lsat.df.shape[0])
    y[ps] = 2
    y[bl] = 1
    y[gr] = 0

    bl_samp = np.random.choice(bl, size=16, replace=False)
    gr_samp = np.random.choice(gr, size=16, replace=False)

    index = np.concatenate((gr, bl)) # include all three or just two?
    X = df.loc[index]
    y = y[index]
    return X, y

def classify_games():
    '''
    Perform a grid search to test various parameterizations, based on choice of 3 data sources:
        1) the text tokens from the puzzle prompts
        2) the text tokens from the rules prompts
        3) the text tokens from the rules POS tags

    Return: lists of the data for all 3 sources
    '''
    balanced, y = balance_classes(Lsat.keyed)
    X_prompts, _ , idx = prep_X_y_classifier(balanced, Lsat.prompts, Lsat.df.primary_type)
    X_rules, _ , idx = prep_X_y_classifier(balanced, Lsat.rules, Lsat.df.primary_type)
    both = [X_prompts[i] + X_rules[i] for i in range(len(X_prompts))]

    print ("starting grid search using rules tags ...")
    grid(X_r_tags, y)

    return X_prompts, X_rules, X_r_tags

def separate_classifiers(train_mask, test_mask):
    '''
    Generate accuracy scores, training set probabilities, and
        test set probabilities for all three data sources used in the first
        layer of the stacked model:
            1) the texts of the prompts,
            2) the texts of the rules,
            3) the texts of the POS tags for the rules

    Return: Accuracy scores for all three sources (prompts, rules, rules tags):
                --p_score, r_score, t_score
            Probabilities associate with the training data for all three sources:
                --prompt_prob, rules_prob, tags_prob
            Probabilities for the test set data for all three sources:
                --prompt_test_prob, rules_test_prob, tags_test_prob
    '''
    prompt = CountVectorizer(max_df=1.0, min_df=0.05, stop_words=None)
    X_vecs = prompt.fit_transform(X_prompts[train_mask])
    prompt_model = SGDClassifier(alpha=0.5, n_iter=5, loss='log')
    prompt_model.fit(X_vecs, y[train_mask])

    X_test_vecs = prompt.transform(X_prompts[test_mask])
    prompt_prob = prompt_model.predict_proba(X_vecs) # probabilities predicted on training set
    prompt_test_prob = prompt_model.predict_proba(X_test_vecs) # probabilities predicted on test set
    prompt_pred = prompt_model.predict(X_test_vecs)
    p_score = accuracy_score(y[test_mask], prompt_pred)

    rules = TfidfVectorizer(max_df=0.7, min_df=0.0, stop_words='english')
    X_vecs = rules.fit_transform(X_rules[train_mask])
    rules_model = SGDClassifier(alpha=0.5, n_iter=3, loss='log')
    rules_model.fit(X_vecs, y[train_mask])

    X_test_rules = rules.transform(X_rules[test_mask])
    rules_prob = rules_model.predict_proba(X_vecs) # probabilities predicted on training set
    rules_test_prob = rules_model.predict_proba(X_test_rules) # probabilities predicted on test set
    rules_pred = rules_model.predict(X_test_rules)
    r_score = accuracy_score(y[test_mask], rules_pred)

    tags = CountVectorizer(max_df=0.3, min_df=0.05, stop_words=None) #
    X_tags = tags.fit_transform(X_r_tags[train_mask])
    tags_model = SGDClassifier(alpha=0.5, n_iter=1, loss='log')
    tags_model.fit(X_tags, y[train_mask])

    X_test_tags = tags.transform(X_r_tags[test_mask])
    tags_prob = tags_model.predict_proba(X_tags) # probabilities predicted on training set
    tags_test_prob = tags_model.predict_proba(X_test_tags) # probabilities predicted on test set
    tags_pred = tags_model.predict(X_test_tags)
    t_score = accuracy_score(y[test_mask], tags_pred)

    return p_score, r_score, t_score,\
           prompt_prob, rules_prob, tags_prob,\
           prompt_test_prob, rules_test_prob, tags_test_prob

def custom_CV(k=10, runs=100):
    '''
    Perform customized Cross-Validation on a stacked model that feeds the probability outputs from
        three separate models into a random forest model, which generates a combined probability.
    Print accuracy scores based on each of the three separate models as well as their combined accuracy.
    '''
    size = 50//k
    samples = np.array(range(50))
    for run in range(runs):
        np.random.shuffle(samples)
        test_mask = samples[:size]
        train_mask = samples[3:]
        to_unpack = separate_classifiers(train_mask, test_mask)
                                # returns probs on TRAINING Xs
        p_score, r_score, t_score, p_probs, r_probs, t_probs, p_pred, r_pred, t_pred = to_unpack
        p_scores[run] = p_score
        r_scores[run] = r_score
        t_scores[run] = t_score
        c_score = ensemble_model(RandomForestClassifier, train_mask, test_mask,\
                                    p_probs[:,1], r_probs[:,1], t_probs[:,1],\
                                    p_pred[:,1], r_pred[:,1], t_pred[:,1])
        c_scores[run] = c_score
    print ("Average score based on prompts alone: ", p_scores.mean(), p_scores.std())
    print ("Average score based on rules alone: ", r_scores.mean(), r_scores.std())
    print ("Average score based on tags alone: ", t_scores.mean(), t_scores.std())
    print ("Average score based on all three: ", c_scores.mean(), c_scores.std())

def ensemble_model(model, train_mask, test_mask,\
                    p_probs, r_probs, t_probs,\
                    p_pred, r_pred, t_pred):
    '''
    Return a combined accuracy score based on the stacked, ensemble model.
    '''
    train_y = y[train_mask]
    train_X = np.hstack((p_probs.reshape(-1, 1), r_probs.reshape(-1, 1), t_probs.reshape(-1, 1)))
    test_X = np.hstack((p_pred.reshape(-1, 1), r_pred.reshape(-1, 1), t_pred.reshape(-1, 1)))
    mod = model()
    mod.fit_transform(train_X, train_y)
    predictions = mod.predict(test_X)
    c_score = accuracy_score(y[test_mask], predictions)
    return c_score

def majority_vote(p_prob, r_prob, t_prob, test_mask):
    '''
    Take the majority vote from 3 different models, based on three different data sources.

    Input: Probabilites produced based on the prompts, rules, and rules tags,
            as well as a mask containing the indices for the test set.
    '''
    predictions = np.zeros(50)
    for i, real in enumerate(test_mask):
        p, r, t = 0, 0, 0
        p_pred, r_pred, t_pred = p_prob[i][1], r_prob[i][1], t_prob[i][1]
        if p_pred>.5:
            p = 1
        if r_pred>.5:
            r = 1
        if t_pred>.5:
            t = 1
        if p + r + t >= 2: # simply majority vote
            predictions[real] = 1
        print ("p_pred {} r pred {} t pred {} c pred {}".format(p_pred, r_pred, t_pred, predictions[real]))
    score = accuracy_score(y[test_mask], predictions[test_mask])
    return score

def collect_rules_tags_by_game(df, rules_tags):
    '''
    Convert lists of POS tags into merged strings.

    Input: DataFrame, list of lists for POS tags for rules
    Return: Merged string, SpaCy doc objects
    '''
    result = []
    for game in rules_tags:
        collection = ''
        for rule in game:
            merged_sent = ' '.join(rule) # merge tags from one rule into one sentence
            collection += merged_sent
        result.append(collection)
    docs = []
    for game in df.iterrows():
        docs.append(result[game[0]])
    return result, docs

def print_labels():
    '''
    Print out different versions of possible seq2seq labels for comparison purposes.
    '''
    for key in label.keys():
        print('-------------------------- {} ---------------------------'.format(key))
        print (label[key])
        print (cond_label[key][0])
        print (xcond_label[key][0])

if __name__ == '__main__':
    print ("loading data...")
    Lsat = load_pickle('LSAT_data')

    print ("loading SpaCy...")
    nlp = spacy.load('en')

    label = load_pickle('rule_labels')
    cond_label = load_pickle('rule_cond_labels')
    xcond_label = load_pickle('rule_xcond_labels')
    #print_labels()
    #xcond_label = {} #initialize new file as empty dictionary
    #save_pickle(xcond_label, 'rule_xcond_labels')

    print ('generating SpaCy docs...')
    gen_spacy_docs(Lsat.keyed, Lsat.prompts, Lsat.prompts_as_spdoc, Lsat.prompts_1_spdoc)
    gen_spacy_docs(Lsat.keyed, Lsat.rules, Lsat.rules_as_spdoc, Lsat.rules_1_spdoc)
    gen_spacy_docs(Lsat.keyed, Lsat.questions, Lsat.questions_as_spdoc, Lsat.questions_1_spdoc)
    gen_spacy_docs_dict(Lsat.keyed, Lsat.answers, Lsat.answers_as_spdoc)

    save_spacydoc(Lsat.prompts_1_spdoc, 'prompt_as_doc')
    save_spacydocs(Lsat.rules_as_spdoc, 'rules_as_docs')
    save_spacydocs(Lsat.questions_as_spdoc, 'questions_as_docs')
    save_spacydocs_dict(Lsat.answers_as_spdoc, 'answers_as_docs')

    prompt_text, prompt_tags, prompt_pairs = load_spacey('prompt_as_doc')
    rules_text, rules_tags, rules_pairs = load_spacey('rules_as_docs')
    questions_text, questions_tags, questions_pairs = load_spacey('questions_as_docs')
    answers_text, answers_tags, answers_pairs = load_spacey('answers_as_docs')

    check_all_SpaCy(Lsat.keyed_seq_lin) # adds labels as needed

    #print ('testing models...')
    balanced, y = balance_classes(Lsat.keyed)
    X_prompts, _ , idx = prep_X_y_classifier(balanced, Lsat.prompts, Lsat.df.primary_type)
    X_rules, _ , idx = prep_X_y_classifier(balanced, Lsat.rules, Lsat.df.primary_type)
    all_r_tags, X_r_tags = collect_rules_tags_by_game(balanced, rules_tags)

    #print ('testing classifier models...')
    #X_prompts, X_rules, X_r_tags = classify_games()

    X_prompts = np.array(X_prompts)
    X_rules = np.array(X_rules)
    X_r_tags = np.array(X_r_tags)

    runs = 100
    p_scores, r_scores, t_scores, c_scores = np.zeros(runs), np.zeros(runs), np.zeros(runs), np.zeros(runs)
    p_prob, r_prob, t_prob = np.zeros((runs, 50)), np.zeros((runs, 50)), np.zeros((runs, 50))
    #custom_CV(runs=runs)
    #predictions = combined_score(p_prob, r_prob)
