class Rule(object):

    '''
    Rule objects to represent logical rules employed in LSAT logic games.
    '''

    def __init__(self, text_list, tags_list, solver, game_num, rule_num=None, question_num=None, answer_num=None):
        self.text_list = text_list # compound variables as single token
        self.text_merged = ' '.join(text_list)
        self.tags_list = tags_list
        self.solver = solver # associated solver object
        self.game_num = game_num
        self.rule_num = rule_num
        self.question_num = question_num
        self.answer_num = answer_num
        self.local = False # default assumes global rule
        self.type = None # no type yet
        self.output = []
        self.local_remainder_type = []
        self.remainder = []
        self.conditions = []
        self.spans = []
        self.rem_span = []

    def parse(self):
        '''
        Convert the English text of a logical rule into Python code.
        '''
        #print ("Creating variables dictionary...")
        self.create_vars_dict()
        #print ("Examining root...")
        self.examine_root()
        #print ("Creating initial pos version of rule...")
        self.convert_rule_to_tags()
        #print ("Checking if negation...")
        self.negation()
        #print ("Bracketing conjunctive booleans.")
        self.bracket_conj_bools()
        #print ("Bracketing comparative booleans.")
        self.bracket_comp_bools()
        #print ("Bracketing negatives.")
        self.bracket_negs()
        #print ("Expanding conjunctive sets.")
        self.expand_all_conjs()
        return self.output # return result of rule parsing

    def create_vars_dict(self):
        '''
        Create a dictionary mapping actual variable names to their generic 'A', 'B', 'C' equivalents,
        such that the rule's first variable becomes 'A', the second 'B', etc.
        That dictionary becomes the attribute vars_dict.
        '''
        self.vars_dict = {}
        letter_idx = ord('A')
        for i, w in enumerate(self.text_list):
            if w in self.solver.vars: # if word is a variable (check solver object)
                if w not in self.vars_dict:
                    self.vars_dict[w] = chr(letter_idx) # populate dict with letters
                    letter_idx += 1

    def _extract_conditions(self):
        '''
        Extract the conditions from a local rule.
        Populate the list attribute conditions_as_Rule_objs[i].
        '''
        conditions = []
        condition = []
        cond_spans = []
        for i, token in enumerate(self.text):
            #print (token)
            if token == 'If':
                start = i + 1
            elif token == 'and':
                stop = i
                span = (start, stop)
                conditions.append(condition) # multiple conditions
                cond_spans.append(span)
                condition = [] # so reset single condition collector for second
                start = stop + 1 # reset new start
                #print ('conditions:', conditions)
                #print ('spans:', spans)
            elif token in ['which', 'then']:
                stop = i
                span = (start, stop)
                conditions.append(condition) # multiple conditions
                cond_spans.append(span)
                break
            else:
                condition.append(token)
        self.remainder = questions_text[game_num][quest_num][i:] # collect rest of question
        self.rem_span = (i, len(self.text_list))
        self.conditions = conditions
        self.cond_spans = cond_spans
        for i, condition in conditions:
            start, stop = cond_spans[i]
            r = Rule(self.text_list[start:stop], self.tags_list[start:stop],\
                     self, self.game_num, self.rule_num)
            self.conditions_as_Rule_objs[i] = r

    def examine_root(self):
        '''
        Examine the root of the sentence to ascertain whether it contains a local rule,
            which only governs a single question, is instead a global rule.
        '''
        if (self.text_list[0] == 'If') or (self.tags_list[0] == 'VB'):
            self.local = True
            self._extract_conditions()
        # if rule is global, no need to populate .remainder, .conditions, etc.

    def convert_rule_to_tags(self):
        '''
        Create customized, quasi-POS tags for rule tokens.
        '''
        label_tags = []
        self.output = []
        self.conj_sets_to_expand = [] # ((start, stop), conj, (start, stop))
        for i, token in enumerate(self.text_list):
            if token in comparitor_words:
                label_tags.append('Comp')
                self.output.append(comparitor_words[token])
            elif token in conjunctions:
                label_tags.append('Conj') if token != 'not' else label_tags.append('not')
                self.output.append(token)
            elif token in self.vars_dict:
                label_tags.append('Var')
                self.output.append(token)
            elif token in ordinals:
                if not 'Comp' in self.output: # if lack > or <, assume equality
                    self.output.append('==')
                    label_tags.append('Comp')
                label_tags.append('Ord')
                self.output.append(str(ordinals[token]))
        self.label_tags = label_tags
        self.reduced = label_tags
        self.reduced_idx = [(i, i) for i, token in enumerate(self.reduced)] # start, stop indices

    def negation(self):
        '''
        Reorder tokens so that 'not' appears outside expression negated.
        '''
        if not 'not' in self.label_tags:
            return # if no negation, nothing to worry about
        new_label_tags = self.label_tags.copy()
        new_output = self.output.copy()
        for i, pos in enumerate(self.label_tags[:-1]):
            if ((pos=='not') and (self.label_tags[i+1]=='Comp')):
                new_label_tags[i-1] = self.label_tags[i]
                new_label_tags[i] = self.label_tags[i-1]
                new_output[i-1] = self.output[i]
                new_output[i] = self.output[i-1]
        self.output = new_output
        self.label_tags = new_label_tags

    def shift(self, idx_list, increment):
        '''
        Shift indices so that attribute .reduced_idx still maps correctly from .reduced to .output/.label_tags lists
        '''
        return [(start+increment, stop+increment) for start, stop in idx_list]

    def insert(self, lst, loc, element, flag=True):
        '''
        Insert a new element into a label and update all affected indices.
        '''
        lst.insert(loc, element)
        new_idx = []
        for start, stop in self.reduced_idx:
            if start >= loc:
                if flag:
                    new_idx.append((loc, loc)) # insert new idx for new element
                flag = False # no longer need to insert index for new element
                start += 1
            if stop >= loc:
                stop +=1
            new_idx.append((start, stop))
        new_conj = []
        for i in self.conj_sets_to_expand:
            before, c, after = i
            if before[0] >= loc:
                before[0] += 1
            if before[1] >= loc:
                before[1] +=1
            if after[0] >= loc:
                after[0] += 1
            if after[1] >= loc:
                after[1] +=1
            if c >= loc:
                c +=1
            new_conj.append((before, c, after))
        self.conj_sets_to_expand = new_conj # want?

    def bracket_conj_bools(self):
        '''
        Add parentheses to sets/Booleans defined by conjunctions.
        '''
        new_label_tags = self.label_tags.copy()
        new_output = self.output.copy()
        for i, pos in enumerate(self.label_tags[:-2]):
            if (pos=='Var') and (self.label_tags[i+1]=='Conj') and\
                    (self.label_tags[i+2] in poss_blocks):
                self.insert(new_output, i+3, ')')
                self.insert(new_output, i, '(')
                self.conj_sets_to_expand.append([[i+1,i+1], i+2, [i+3,i+3]])
                self.reduced = self.label_tags[:i] + ['Set'] + self.label_tags[i+3:]
                self.reduced_idx = self.reduced_idx[:i] + [[i, i+4]] + self.shift(self.reduced_idx[i+3:], 2)
        self.output = new_output

    def bracket_comp_bools(self):
        '''
        Add parentheses to Booleans containing comparatives.
        '''
        new_output = self.output.copy()
        for i, pos in enumerate(self.reduced[:-2]):
            if self.reduced[i+1]=='Comp':
                self.insert(new_output, self.reduced_idx[i+2][1]+1, ')', flag=False) # add outside following var/block
                self.insert(new_output, self.reduced_idx[i][0], '(', flag=False) # add outside preceding var/block
        self.output = new_output

    def bracket_negs(self):
        '''
        Add parentheses around negated expressions.
        '''
        new_output = self.output.copy()
        flag = False
        for i, pos in enumerate(self.output):
            if (pos=='not'):
                self.insert(new_output, i, '(')
                flag = True
            elif (flag and (pos==')')):
                self.insert(new_output, i+1, ')')
        self.output = new_output

    def expand_all_conjs(self):
        '''
        Expand expressions containing conjunctions.
        '''
        if not self.conj_sets_to_expand:
            return
        expanded = []
        for expansion in self.conj_sets_to_expand:
            first_idx, conj, second_idx = expansion
            first_half = self.output.copy()
            second_half = self.output.copy()
            del first_half[second_idx[0]:second_idx[1]+1]
            del first_half[conj]
            del second_half[conj]
            del second_half[first_idx[0]:first_idx[1]+1]
            expanded += ['('] + first_half + [self.output[conj]] + second_half + [')']
        self.output = expanded

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
            'must be': 'place',
            'could be': 'place',
            'cannot be': 'place',
            'rule substitution': 'rule',
            'accurate list': 'set',
            'could be completely determined': 'rule',
            'how many': 'number'
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
