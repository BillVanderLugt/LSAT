import spacy
import pickle
from load_categories_df import LSAT
import re
from string import punctuation
from collections import Counter
import itertools
from rule_class import Rule

from pprint import pprint
from time import time
import logging
import pandas as pd
import numpy as np

from feature_engineering import save_pickle, load_pickle, \
                            save_spacydoc, save_spacydocs, load_spacey

class Solver(object):

    '''
    A Solver object gets created for each puzzle to be solved.
    The Solver includes 7 main methods, associated with the 7 solving steps:
        step_1: Identify the Puzzle Type
        step_2: Set Up the Puzzle
            -Extracting the Variable Names
            -Populating Their Permutations
        step_3: Parse the Puzzle's Rules
        step_4: Apply All of the Rules
            -Winnow the Pool of Permissible Solutions
        step_5: Parse the Questions
        step_6: Parse the Answers
        step_7: Pick the Correct Answer
    '''

    def __init__(self, game_num=3):
        self.game_num = game_num # which game (by index number in Lsat.df) to solve
        self.type = '' # Sequencing or Basic Linear for now
        self.prompt_tags = prompt_tags[game_num] # pull out prompt tags
        self.prompt_text = prompt_text[game_num]
        self.prompt_pairs = prompt_pairs[game_num]
        self.rules_tags = rules_tags[game_num] # pull out rule tags
        self.rules_text = rules_text[game_num]
        self.rules_pairs = rules_pairs[game_num]

        self.referenced_vars = {} # dictionary of key = (rule_no, word_no)
                                        #         val = variable string
        ### FOR LOCAL RULES ###
        self.answer_type = [] # list of answer types
        self.local_rules = {} # key = question number--expand for answers w local rules?
        self.remainder = {}
        self.conditions = {}
        self.spans = {}
        self.rem_span = {}
        self.winnowed_by_local = {}

    def step_1(self):
        '''
        STEP 1: Identify the Puzzle Type
        My stacked model predicts with about 93 percent accuracy.
        I have not incorporated that model here because I would still like to experiment with CNN solutions
            and, more practically, because the remainder of the project is limited to one particular puzzle type (sequencing puzzles),
            the remaining steps simply assume type is 'Pure Sequencing'.
        '''
        self.type = 'Pure Sequencing'
        return self.type

    def _find_first(self):
        '''
        Search through the part of speech tags from the puzzle prompt to locate
            the first variable from a list of comma-separated variables (usually 6-9 variables).
        For confirmation, must search ahead to the second and third items in the series.
        Must be able to handle compound variables consisting of multiple nouns.

        Return: the index number of the first variable
        '''

        for i, pos in enumerate(self.prompt_tags):
            if pos in poss: # look for first possible var pos
                if (self.prompt_tags[i+1] == ',') or (self.prompt_tags[i+2] == ','): # check for comma after it
                    for j in range(2, 5): # check for comma within 3 four words
                        if self.prompt_tags[i+j] == ',': # look for second comma
                            if self.prompt_tags[i+j-1] in poss: # confirm second comma follows noun
                                for k in range(2, 5): # check for a third comma within 3 four words
                                    if self.prompt_tags[i+j+k] == ',': # look for third comma
                                        if self.prompt_tags[i+j+k-1] in poss: # confirm third comma follows noun
                                            return i # sequence is confirmed so return index of original noun
            continue # if noun not followed by a comma, continue to next noun
        print ("ERROR: Can't find first!")
        print (self.game_num, self.prompt_pairs)
        return None # couldn't find it = error

    def _collect_vars(self, i):
        '''
        Collect the variables from the puzzle prompt.

        Input i, the index of the first variable.
        Return a list of the variables upon which the puzzle is based.
        '''

        vars = [] # intialize the list to populate and return
        var = '' # for compound variables, collect them as a single token
        while i < (len(self.prompt_tags)-1):
            if self.prompt_tags[i] in poss: # look for next possible variable
                var += self.prompt_text[i] # collect in case compound variable
                if (self.prompt_tags[i+1] == ',') or (self.prompt_tags[i+1]=='and'):
                    vars.append(var) # sequence is confirmed, so noun is var
                    i += 2
                    var = ''
                elif self.prompt_tags[i+1] in ':.':
                    vars.append(var) # noun ends the series
                    #print ('variables ', vars)
                    return vars
                else: # must be compound noun so just increment to see if next is also noun
                    i += 1
                    var += ' ' # compound variables as single tokens
            else:
                i += 1
        return vars

    def step_2(self):
        '''
        STEP 2: Set-Up Puzzle
            - Extract Variable Names
            - Populate Event Space
        '''

        first = self._find_first()
        i = first
        # print ('first var is ', self.prompt_text[first])
        # print ('PROMPT', self.prompt_pairs)
        # print ()
        self.vars = self._collect_vars(i)
        #print ('generating permutations...')
        perms = list(itertools.permutations(self.vars))
        print ('Possible permutations of {} variables: {}'.format(len(self.vars), len(perms)))
        self.permutations = perms # before any rules
        self.viable = perms # viable candidates after rule(s)
        #print ()
        return self.vars

    def step_3(self):
        '''
        STEP 3: Parse a Single Rule
        '''
        print ('running new Step 3 parser with Rule objects...')
        self.rules = [] # list of Rule objects
        for i, txt in enumerate(rules_text[self.game_num]):
            tags_list = rules_tags[self.game_num][i]
            self.rules.append(Rule(txt, tags_list, self, self.game_num, rule_num=i))
        for rule in self.rules:
            rule.parse() # parse each rule

    def _test_item(self, rule, item):
        '''
        Determine whether the ordering of one item/permutation complies with one rule.

        Input: --a list containing the elements of a rule,
               --a list of variables for the permutation/item to test
        Return: a Boolean -- whether the item complies with the rule
        '''
        new_rule = rule.copy()
        item_set = set(item)
        for i, word in enumerate(rule):
            if word in item_set:
                idx = item.index(word)
                new_rule[i] = str(idx)
        new_as_string = "".join(new_rule)
        result = eval(new_as_string)
        if (np.random.rand() < 0.00001):
            print ("           random sample: ", new_as_string, result)
        return result

    def _winnow_one(self, rule, pool):
        '''
        Shrink the pool of candidates to those that comply with the rule.

        Input: the rule as a list of elements, the pool as a list of permutations (as tuples)
        Return: a winnowed list of all the candidates from the pool that passed the test
        '''
        shrunk = [item for item in pool if self._test_item(rule, item)]
        return shrunk

    def _winnow_all(self):
        '''
        Apply all the rules to the pool of conceivable permutations.
        Winnow out those that satisfy all the rules.

        Return: the winnowed pool as a list
        '''
        pool = self.viable
        print ("With {} variables, the pool starts with {} permutations...".format(len(self.vars), len(pool)))
        for i, rule in enumerate(self.rules):
            print ("    ", " ".join(rule.text_list))
            pool = self._winnow_one(rule.output, pool)
            print ("After rule {} pool size shrunk down to {}".format(i, len(pool)))
            print ()
        if len(pool)<10: # if the remaining pool is small, go ahead and print it
            print(pool)
        return pool

    def step_4(self):
        '''
        STEP 4: Apply Rules to Winnow Solutions
        '''
        self.winnowed = self._winnow_all()

    def _scan_for_ordinals_cardinals(self, tokens):
        '''
        Scan tokens to extract any place numbers, including references to the 'last' variable.

        Input: a list of tokens
        Return: the places refered to (reindexed to start at zero: 'first'--> 0 )
        '''
        results = []
        for token in tokens:
            if token in ordinals:
                if token == 'last': # if looking for last place
                    results.append(len(self.vars)-1) # last = number of variables (zero indexing)
                else:
                    results.append(ordinals[token]) # otherwise just convert ordinal to cardinal
            elif token.isdigit(): # include any cardinals
                results.append()
        return results

    def _get_q_type(self, game_num, quest_num):
        '''
        Collect the parameters required to ascertain a question's type.
        Hand off to a helper function that actually performs the identification.

        Input: the puzzle/game's number, the number of the question within that game
        Return: a quest_type tuple and a list of the referenced place numbers
        '''
        sent = Lsat.questions[game_num][quest_num] # question as single string
        tokens = questions_text[game_num][quest_num] # question as list of tokens
        first_word = tokens[0] # first token
        pos_list = questions_tags[game_num][quest_num] # list of tags
        cardinals = self._scan_for_ordinals_cardinals(tokens)
        return self._step_5_helper(sent, tokens, first_word, pos_list, cardinals, game_num, quest_num)

    def _step_5_helper(self, sent, tokens, first_word, pos_list, cardinals, game_num, quest_num):
        '''
        Ascertain a question's type, including whether it includes a local rule or just a global rule.

        Input: a variety of parameters
        Return: the question type information (as a tuple) and a list of place numbers
        '''
        if (first_word == 'If') or (pos_list[0] == 'VB'): # two versions of local rule cue
            quest_type = ['local']
            self._local_rules_handler(game_num, quest_num)

        else:
            quest_type = ['global']
        if (sent.find('could be the order') > 0) or (sent.find('could be an accurate') > 0):
            quest_type.append('could be the order') # 'acceptability' question

        elif (sent.find('must be true')> 0) and (sent.find('EXCEPT')==-1):
            quest_type.append('must be true')
        elif (sent.find('CANNOT be false')> 0) and (sent.find('EXCEPT')==-1):
            quest_type.append('must be true')
        elif (sent.find('could be false')> 0) and (sent.find('EXCEPT')>0):
            quest_type.append('must be true')

        elif (sent.find('CANNOT be true') > 0) and (sent.find('EXCEPT')==-1):
            quest_type.append('cannot be true')
        elif (sent.find('must be false') > 0) and (sent.find('EXCEPT')==-1):
            quest_type.append('cannot be true')
        elif (sent.find('could be true') > 0) and (sent.find('EXCEPT')>0):
            quest_type.append('cannot be true')

        elif (sent.find('could be true') > 0) and (sent.find('EXCEPT')==-1):
            quest_type.append('could be true')

        elif (sent.find('could be false')>0) and (sent.find('EXCEPT')==-1):
            quest_type.append('could be false')
        elif (sent.find('must be true')> 0) and (sent.find('EXCEPT')>0):
            quest_type.append('could be false')

        ### QUESTIONS DEALING WITH SPECIFIC LOCATIONS ###

        elif cardinals: # does question reference a specific location?

            if (sent.find('must be')> 0) and (sent.find('EXCEPT')==-1):
                quest_type.append('must be')
            elif (sent.find('could be')> 0) and (sent.find('EXCEPT')>0):
                quest_type.append('must be')

            elif (sent.find('CANNOT be') > 0) and (sent.find('EXCEPT')==-1):
                quest_type.append('cannot be')
            elif (sent.find('must be') > 0) and (sent.find('EXCEPT')>0):
                quest_type.append('cannot be')

            elif (sent.find('could be') > 0) and (sent.find('EXCEPT')==-1):
                quest_type.append('could be')
            elif (sent.find('CANNOT be')>0) and (sent.find('EXCEPT')>0):
                quest_type.append('could be')

        ### UNUSUAL QUESTION TYPES ###
        elif (sent.find('if substituted')>0):
            quest_type.append('rule substitution')
        elif (sent.find('accurate list')>0):
            quest_type.append('accurate list')
        elif (sent.find('could be completely determined')>0):
            quest_type.append('could be completely determined')
        elif (sent.find('how many')>0):
            quest_type.append('how many')
        else:
            quest_type.append('unidentified')

        return quest_type, cardinals

    def step_5(self, game_num, quest_num):
        '''
        STEP 5: Parse Question Type
        '''
        quest_type, cardinals = self._get_q_type(game_num, quest_num)
        return quest_type, cardinals

    def _local_rules_handler(self, game_num, quest_num):
        '''
        Handle questions that are locally conditioned (only for one question) on an additional rule.

        Input: the game number and question number
        Return: None--just populates a whole series of special local rule attributes
        '''

        conditions, spans, remainder, rem_span = self._extract_conditions(game_num, quest_num)
        local_rules = [] # collect local Rules objects
        for i, condition in enumerate(conditions):
            try:
                text_list = condition
                tags_list = questions_tags[game_num][quest_num][spans[i][0]:spans[i][1]] # slice out pos_list
                local_rules.append(Rule(text_list, tags_list, self, game_num, question_num=quest_num))
                local_rules[i].parse()
            except:
                print ('Error handling local rule...')
        q_type = self._get_local_q_type(remainder, rem_span, game_num, quest_num)
        self.local_rules[quest_num] = local_rules
        self.conditions[quest_num] = conditions
        self.spans[quest_num] = spans
        self.remainder[quest_num] = remainder
        self.rem_span[quest_num] = rem_span

        ## APPLY LOCAL RULES TO POOL ##
        print ('Applying local rules to winnowed pool...')
        pool = self.winnowed
        print ('INITIAL POOL SIZE = ', len(pool))
        for i, rule in enumerate(local_rules):
            pool = self._local_winnow(rule, pool)
            print ('ROUND {} WINNOWED POOL SIZE = {}'.format(i, len(pool)))
        self.winnowed_by_local[(game_num, quest_num)] = pool

    def _extract_conditions(self, game_num, quest_num):
        '''
        Convert the conditions contained in the local rules to Rule objects.

        Input: game number, question number
        Returns: -- conditions: a list containing Rule objects for each of the local rules
                 -- spans: (start, stop) indices identifying the assocated spans from the question
                 -- remainder: the remaining question stem that following the local rule conditions
                 -- rem_span: (start, stop) indices identifying the span of the remainding question stem
        '''

        conditions = []
        condition = []
        spans = []
        start = 0
        for i, token in enumerate(questions_text[game_num][quest_num]):
            if token == 'If':
                start = i + 1
            elif token == 'and':
                stop = i
                span = (start, stop)
                conditions.append(condition) # multiple conditions
                spans.append(span)
                condition = [] # so reset single condition collector for second
                start = stop + 1 # reset new start
                #print ('conditions:', conditions)
                #print ('spans:', spans)
            elif token in ['which', 'then']:
                stop = i
                span = (start, stop)
                conditions.append(condition) # multiple conditions
                spans.append(span)
                break
            else:
                condition.append(token)
        remainder = questions_text[game_num][quest_num][i:] # collect rest of question
        rem_span = (i, len(questions_text[game_num][quest_num]))
        return conditions, spans, remainder, rem_span

    def _get_local_q_type(self, remainder, span, game_num, quest_num):
        '''
        Ascertain the type of the question that remains after the local rules.
        Perform the same analysis on the remaining question stem as one does on a global rule.

        Input: --remainder: list of tokens
               --span: (start, stop) index to the original question (to access Parts of Speech)
               --game number
               --question number
        Return: question type tuple and list of place numbers
        '''
        sent = " ".join(remainder) # question as single string
        tokens = questions_text[game_num][quest_num][span[0]:span[1]] # question as list of tokens
        first_word = tokens[0] # first token
        pos_list = questions_tags[game_num][quest_num][span[0]:span[1]] # list of tags
        cardinals = self._scan_for_ordinals_cardinals(tokens)
        return self._step_5_helper(sent, tokens, first_word, pos_list, cardinals, game_num, quest_num)

    def _local_winnow(self, rule, pool):
        '''
        Apply the local rules to further winnow the pool of candidate permutations.

        Input: a rule and a pool
        Return: the winnowed pool
        '''
        print ("The pool currently contains {} permutations.".format(len(pool)))
        print ("    before applying ", " ".join(rule.output))
        pool = self._winnow_one(rule.output, pool)
        print ("Now pool size has shrunk down to {}".format(len(pool)))
        if len(pool)<20:
            pprint(pool)
        return pool

    def _process_local_rule(self, game_num, quest_num):
        pool = self.winnowed_by_local[(game_num, quest_num)]
        q_type = self.quest_types[quest_num]
        return q_type[0][1] # route Step 6 to appropriate method for expected answer format, given question type

    def step_6(self, game_num, quest_num):
        '''
        STEP 6: Parse Answersq_
        '''
        q_type = self.quest_types[quest_num]
        if q_type[0][0] == 'local':
            print ('OK, is local so routing to _process_local_rule...')
            key = self._process_local_rule(game_num, quest_num)

        key = q_type[0][1]
        return expect[key] # return expected answer format, given question type

    def step_7(self, game_num, quest_num):
        '''
        STEP 7: Select Correct Answer
        '''
        correct = 'dunno'
        router = self.answer_type[quest_num]
        quest_type = self.quest_types[quest_num]
        #print ('Ready to answer {} type question with {} answer'.format(quest_type, router))

        if router == 'whole sequence':
            correct = self._step_7_whole_seq(game_num, quest_num, quest_type)
        elif router == 'statement':
            correct = self._step_7_statement(game_num, quest_num, quest_type)
        elif router == 'place':
            correct = self._step_7_place(game_num, quest_num, quest_type)
        elif router == 'rule':
            correct = self._step_7_rule(game_num, quest_num, quest_type)
        elif router == 'set':
            correct = self._step_7_set(game_num, quest_num, quest_type)
        elif router == 'number':
            correct = self._step_7_number(game_num, quest_num, quest_type)
        else:
            print ("problem routing to correct answer format")
        print ("        CORRECT ANSWER: ", correct)
        return correct

    def _step_7_parse_seq(self, game_num, quest_num):
        '''
        For answers that consists of sequences of variables, parse the sequences.

        Input: game number and question number
        Return: a list of lists (sequences) for each of the 5 possible answers.
        '''

        answers = []
        ans_list = Lsat.answers[game_num][quest_num]
        print ('ans_list ', ans_list)
        for answer in ans_list:
            #print (answer)
            cleaned = [t.lstrip() for t in answer.split(',')]
            for t in cleaned:
                if not t in self.vars:
                    print ("Ruh, roh, I don't recognize ", t)
            answers.append(cleaned)
        return answers

    def _step_7_whole_seq(self, game_num, quest_num, quest_type):
        '''
        Determine which answer is correct.

        Input: game number, question number, question type
        Return: a string identifying the correct answer (including its letter)
        '''
        answers = self._step_7_parse_seq(game_num, quest_num)
        correct = ''
        for i, answer in enumerate(answers):
            #print ('testing answer:', answer)
            if tuple(answer) in self.winnowed:
                correct = chr(65+i) + ': ' + ", ".join(answer)
        if not correct:
            print ("Ruh, roh, I couldn't find a viable answer in winnowed pool!")
        return correct

    def gen_spacy_doc_answer(self, sent):
        '''
        Use SpaCy to turn an answer into a SpaCy doc object, which includes the parts of speech as tag_ attributes.

        Input: entire sentence as a single string
        Return: a list of the text tokens, a list of the POS tokens
        '''

        doc = nlp(sent) # generate SpaCy doc object
        doc = [w for w in doc if w.tag_ != 'POS'] # eliminate possessives
        ans_tags = [w.tag_ for w in doc]
        ans_text = [w.text for w in doc]
        return ans_text, ans_tags

    def _editor(self, line):
        '''
        Because SpaCy parses a single-letter variable or number followed by a period as an abbreviation,
            an extra space must be added to clarify the syntax in such cases.

        Input: a token
        Return: the same token with an extra space added if required
        '''

        if len(line) < 2:
            return line
        elif (line[-1]=='.'):
            if line[-2].isupper():
                line = line[:-1] + ' .'
            if line[-2].isdigit():
                line = line[:-1] + ' .'
        return line

    def _step_7_parse_statements(self, game_num, quest_num):
        '''
        Convert an answer that takes the form of a statement into a Rule object.

        Input: game number, question number
        Return: the tokenized statement
        '''

        answers = []
        ans_as_rule_output = []
        ans_list = Lsat.answers[game_num][quest_num]
        for i, answer in enumerate(ans_list):
            text_list, tags_list = self.gen_spacy_doc_answer(self._editor(answer))
            rule = Rule(text_list, tags_list, self, game_num, question_num=quest_num, answer_num=i)
            output = rule.parse()
            ans_as_rule_output.append(rule.output)
        return ans_as_rule_output

    def _step_7_statement(self, game_num, quest_num, quest_type):
        '''
        Handle answers that take the form of a statement.

        Input: game number, question number, question type
        Return: test of correct answer, including letter number
        '''

        outputs = self._step_7_parse_statements(game_num, quest_num)
        correct = ''
        type = quest_type[0][1]
        if quest_type[0][0]=='local':
            print ('OK, question has local rule(s) so using winnowed pool...')
            pool = self.winnowed_by_local[(game_num, quest_num)]
        else:
            pool = self.winnowed
        print ('initial pool size: ', len(pool))
        print ('initial pool: ', pool)

        if type == 'must be true':
            for i, answer_sent in enumerate(Lsat.answers[game_num][quest_num]):
                passed = self._winnow_one(outputs[i], pool)
                if len(passed) == len(pool): # rule holds true for all == correct
                    print ('this rule holds true for everything in pool')
                    correct += chr(65+i) + ': ' + answer_sent
        elif type == 'could be true':
            for i, answer_sent in enumerate(Lsat.answers[game_num][quest_num]):
                passed = self._winnow_one(outputs[i], pool)
                if passed: # rule holds true for some == correct
                    print ('this rule holds true for something in pool')
                    correct += chr(65+i) + ': ' + answer_sent
        elif type == 'could be false':
            for i, answer_sent in enumerate(Lsat.answers[game_num][quest_num]):
                passed = self._winnow_one(outputs[i], pool)
                if len(passed) != len(pool): # rule holds true for some == correct
                    print ('this rule holds true for something in pool')
                    correct += chr(65+i) + ': ' + answer_sent
        elif type == 'cannot be':
            for i, answer_sent in enumerate(Lsat.answers[game_num][quest_num]):
                passed = self._winnow_one(outputs[i], pool)
                if not passed: # rule holds true for some == correct
                    print ('this rule holds true for nothing in pool')
                    correct += chr(65+i) + ': ' + answer_sent
        return correct

        if not correct:
            print ("Ruh, roh, I couldn't find a viable answer statement!")
        return correct

    def _allowed_in_place(self, place):
        allowed = set()
        for perm in self.winnowed:
            if perm[place] not in allowed: # if new variable appears, add to set of allowable
                allowed.add(perm[place])
        return allowed

    def _step_7_place(self, game_num, quest_num, quest_type):
        place = quest_type[1][0]
        type = quest_type[0][1]
        allowed = self._allowed_in_place(place)

        correct = 'dunno'
        if type == 'must be':
            if len(allowed)>1:
                print ("Yikes, I saw multiple options for that position:", allowed)
        elif type == 'could be':
            for i, answer in enumerate(Lsat.answers[game_num][quest_num]):
                if answer in allowed:
                    correct = chr(65+i) + ': ' + answer
        elif type == 'cannot be':
            for i, answer in enumerate(Lsat.answers[game_num][quest_num]):
                if not answer in allowed:
                    correct = chr(65+i) + ': ' + answer
        return correct

    def _step_7_rule(self, game_num, quest_num, quest_type):
        pass

    def _step_7_set(self, game_num, quest_num, quest_type):
        pass

    def _step_7_number(self, game_num, quest_num, quest_type):
        pass


def print_games_w_keyed_questions():
    for i, q in enumerate(Lsat.questions):
        if q:
            print ()
            print ('Game {} has these questions: '.format(i))
            pprint (q)

if __name__ == '__main__':

    poss = ['NN', 'NNP', 'NNPS', 'NNS', 'CD', 'VBG'] # possible pos for variables
    comparitors = ['JJR', 'IN', 'RBR'] # pos that can function as comparitors
    ordinals = {'first': 0,
                'second': 1,
                'third': 2,
                'fourth': 3,
                'fifth': 4,
                'sixth': 5,
                'seventh': 6,
                'eighth': 7,
                'ninth': 8,
                '1': 0,
                '2': 1,
                '3': 2,
                '4': 3,
                '5': 4,
                '6': 5,
                '7': 6,
                '8': 7,
                '9': 8,
                'last': '*'}

    expect = {  'could be the order': 'whole sequence',
                'must be true': 'statement',
                'could be true': 'statement',
                'cannot be true': 'statement',
                'could be false': 'statement',
                'must be': 'place',
                'could be': 'place',
                'cannot be': 'place',
                'rule substitution': 'rule',
                'accurate list': 'set',
                'could be completely determined': 'rule',
                'how many': 'number',
                'unidentified': 'unidentified'
                }

    comparitor_words = {'more': '<',
                        'less': '>',
                        'before': '<',
                        'after': '>',
                        'greater': '>',
                        'higher': '>',
                        'lower': '<',
                        'closer': '<'}

    conjunctions = {'and': '|',
                    'or': '&',
                    'but': '^',
                    'not': 'not'}

    poss_blocks = set(('Var', 'Ord', 'Set'))

    keepers = set(ordinals) | set(conjunctions) | set(comparitor_words)

    Lsat = load_pickle('LSAT_data')
    prompt_text, prompt_tags, prompt_pairs = load_spacey('prompt_as_doc')
    rules_text, rules_tags, rules_pairs = load_spacey('rules_as_docs')
    questions_text, questions_tags, questions_pairs = load_spacey('questions_as_docs')
    answers_text, answers_tags, answers_pairs = load_spacey('questions_as_docs') # update feature_engineering?
    label = load_pickle('rule_labels')

    all_seq = Lsat.keyed_seq.index

### CREATE SET OF PUZZLES TO SOLVE ###
    sf = []
    simple_five = [3, 13, 41, 186, 239]
    programs = [145] # debug reversal, 3 statement answers
    rowers = [186] # matching, 4 statement answers, how many
    activities = [245] # compound variables, immediately before/after
    which_games = programs
    for game_num in which_games:
        #print ('working on game ', game_num)
        sf.append(Solver(game_num)) # populate sf list with 5 Solver objects

### STEP 1 ###
    print ()
    print ("   ---- STEP 1: CATEGORIZING GAMES ---- ")
    for s in sf:
        s.step_1() # ascertain type of each game -- always 'Pure Sequence' for now
    # var_dicts = []

### STEP 2 ###
    print ()
    print ("   ---- STEP 2: EXTRACTING VARIABLES FROM PROMPT ---- ")
    for i, s in enumerate(sf):
        #vars.append(s.step_2()) # get dictionary of variables for each game
        #print (which_games[i])
        vars = s.step_2()
        # print (vars, len(vars))
        # print (s.prompt_text)
        #print ()

### STEP 3 ###
    print ()
    print ("   ---- STEP 3: PARSING RULES ---- ")
    for s in sf:
        s.step_3()
    #    s.sidestep_3()
    #print ()

# ## STEP 4 ###
    print ()
    print ("   ---- STEP 4: APPLYING RULES TO SOLVE PUZZLES ---- ")
    for s in sf: # limit to last puzzle
        print ()
        print ('GAME #: ', s.game_num)
        s.step_4()
    # s = sf[1]
    # s.step_4()

    #print_games_w_keyed_questions()

### STEP 5 ###
    print ()
    print (' ---- STEP FIVE: PARSING QUESTIONS ---- ')
    for s in sf:
        quest_types = []
        for i, q in enumerate(Lsat.questions[s.game_num]): # iterate through each question
            print ("Parsing this question: ", q)
            #print ("Tagged question: ", questions_tags[s.game_num][i])
            quest_type, cardinals = s.step_5(s.game_num, i)
            print ('Step 5 believes this question is: ', quest_type)
            if cardinals:
                print ('And found these numbers: ', cardinals)
            quest_types.append((quest_type, cardinals))
            print ()
        s.quest_types = quest_types

### STEP 6 ###
    print ()
    print (' ---- STEP SIX: PARSING ANSWERS ---- ')
    for s in sf:
        answer_types = []
        for i, question in enumerate(Lsat.questions[s.game_num]): # iterate through each answer
            print ()
            print ('parsing question ', i)
            answer_type = s.step_6(s.game_num, i)
            answer_types.append(answer_type)
        s.answer_type = answer_types

### STEP 7 ###
    print ()
    print (' ---- STEP SEVEN: PICK CORRECT ANSWER ---- ')

    print ("loading SpaCy...")
    nlp = spacy.load('en')

    for s in sf:
        predicted_answers = []
        for i, question in enumerate(Lsat.questions[s.game_num]): # iterate through each question
            print ()
            print ('answering question ', i, question)
            print ('question type', s.quest_types[i])
            predicted = s.step_7(s.game_num, i) # ascertain predicted answer
            predicted_answers.append(predicted)
        s.predicted_answer = predicted_answers
