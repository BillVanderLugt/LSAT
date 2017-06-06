import spacy
import pickle
from load_categories_df import LSAT
import re

# class Game(object):
#
#     def __init__(self):
#     self.prompt = []
#     self.rules = []
#
#     def load(self, number):
#     self.prompt = self.prompt

def load_pickle(name):
    '''
    returns unpickled file
    '''
    with open("../data/" + name + ".pkl", 'rb') as f_un:
        file_unpickled = pickle.load(f_un)
    print ("done unpickling ", name)
    return file_unpickled

def clean_sent(sent):
    return " ".join([clean_word(w) for w in sent.split(' ') if w[0].isalnum()])

def clean_word(word):
    '''
    eliminates parentheses
    '''
    new = re.sub(r'[()]', '', word)
    if new != word:
        print ("@@@@@@ swapping in {} for {} @@@@@".format(new, word))
    return new

def sent_pos(sent):
    cleaned = clean_sent(sent)
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
    cleaned_as_list = cleaned.split(' ')
    print ("Len of original: {}   Len of parsed sans punct: {}".\
                    format(len(cleaned_as_list), len(out)))
    if len(cleaned_as_list) != len(out):
        print ("$$$$$$$$$$$$$$$$$$$$$ ERROR $$$$$$$$$$$$$$$$$$$$$$$")
        for i, w in enumerate(cleaned_as_list):
            print (w, out[i])
    return out, out_plus_punct

def _print_parse_sent(sent):
    # for word in sent.split(' '):
    #     print (word.ljust(12), end='')
    # print (end='\n')
    parts_of_speech, pos_plus_punct = sent_pos(sent)
    # for word in parts_of_speech:
    #     print (word.ljust(12), end='')
    # print (end='\n\n')
    return parts_of_speech, pos_plus_punct

def print_parse(df, source, destination_list):
    for game in df.iterrows():
        print ('############## processing game #: {} #################'.format(game[0]))
        output = []
        for sent in source[game[0]]:
            pos, pos_plus_punct = _print_parse_sent(sent)
            output.append(pos)
        destination_list[game[0]] = output
    return output

def scan_for_term(df, source, term):
    for game in df.iterrows():
        #print ('############## processing game #: {} #################'.format(game[0]))
        for sent in source[game[0]]:
            if term in sent:
                print ("Found {} in game {}".format(term, game[0]))

if __name__ == '__main__':
    Lsat = load_pickle('LSAT_data')
    print ('loading spacy...')
    nlp = spacy.load('en')

    print_parse(Lsat.keyed_seq, Lsat.rules, Lsat.rules_pos_as_list)
    print_parse(Lsat.keyed_seq, Lsat.prompts, Lsat.prompts_pos_as_list)

    scan_for_term(Lsat.keyed_seq, Lsat.rules, 'than')
