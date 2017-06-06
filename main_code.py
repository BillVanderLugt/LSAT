import spacy
from textpipeliner import PipelineEngine
from textpipeliner.pipes import *
import pickle
from load_categories_df import LSAT
import re
from string import punctuation

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

if __name__ == '__main__':
    Lsat = load_pickle('LSAT_data')
    print ('loading spacy...')
    nlp = spacy.load('en')

    #first, second = split_compounds('dummy sentences suck')
    # print (first)
    # print (second)

    #new, paren = _extract_paren('This is a test (dummy) sentence with a (second dummmy) too.')
    extract_parentheticals(Lsat.keyed_seq, Lsat.prompts, Lsat.prompts_as_list)

    tag(Lsat.keyed_seq, Lsat.prompts, Lsat.prompts_as_list, Lsat.prompts_pos_as_list, Lsat.prompts_pos_plus_punct)
    tag(Lsat.keyed_seq, Lsat.rules, Lsat.rules_as_list, Lsat.rules_pos_as_list, Lsat.rules_pos_plus_punct)

    print ('post-check')
    check_tagging(row_num=228)
    #
    # scan_for_term(Lsat.keyed_seq, Lsat.rules, 'than')

########################################################################

########################################################################

pipes_structure = [SequencePipe([FindTokensPipe("VERB/nsubj/NNP"),
                                 NamedEntityFilterPipe(),
                                 NamedEntityExtractorPipe()]),
                       AggregatePipe([FindTokensPipe("VERB"),
                                      FindTokensPipe("VERB/xcomp/VERB/aux/*"),
                                      FindTokensPipe("VERB/xcomp/VERB")]),
                       AnyPipe([FindTokensPipe("VERB/[acomp,amod]/ADJ"),
                                AggregatePipe([FindTokensPipe("VERB/[dobj,attr]/NOUN/det/DET"),
                                               FindTokensPipe("VERB/[dobj,attr]/NOUN/[acomp,amod]/ADJ")])])
                                                                ]

pipes_structure_comp = [SequencePipe([FindTokensPipe("VERB/conj/VERB/nsubj/NNP"),
                                 NamedEntityFilterPipe(),
                                 NamedEntityExtractorPipe()]),
                   AggregatePipe([FindTokensPipe("VERB/conj/VERB"),
                                  FindTokensPipe("VERB/conj/VERB/xcomp/VERB/aux/*"),
                                  FindTokensPipe("VERB/conj/VERB/xcomp/VERB")]),
                   AnyPipe([FindTokensPipe("VERB/conj/VERB/[acomp,amod]/ADJ"),
                            AggregatePipe([FindTokensPipe("VERB/conj/VERB/[dobj,attr]/NOUN/det/DET"),
                                           FindTokensPipe("VERB/conj/VERB/[dobj,attr]/NOUN/[acomp,amod]/ADJ")])])
                                                                ]
def split_compounds(sent):
    '''
    trying out mini-library: https://github.com/krzysiekfonal/textpipeliner
    '''

    sent = nlp(u"The Empire of Japan aimed to dominate Asia and the " \
               "Pacific and was already at war with the Republic of China " \
               "in 1937, but the world war is generally said to have begun on " \
               "1 September 1939 with the invasion of Poland by Germany and " \
               "subsequent declarations of war on Germany by France and the United Kingdom. ")

    engine = PipelineEngine(pipes_structure, sent, [0,1,2])
    engine2 = PipelineEngine(pipes_structure_comp, sent, [0,1,2])
    first_half = engine.process()
    second_half = engine2.process()
    return first_half, second_half
