import pandas as pd
import pickle

class LSAT(object):
    '''
    Create an LSAT class to contain the data from actual LSAT tests.
    '''

    def __init__(self):
        self.prompts = []
        self.rules = []
        self.questions = []
        self.answers = []
        self.prompts_pos = []
        self.rules_pos = []
        self.questions_pos = []
        self.answers_pos = []

    def load(self, file):
        '''
        Load game categorizations from text file.

        Input: file to load_pickle
        Return: a array, ready for conversion into a pandas DataFrame
        '''

        array = []
        with open(file) as f:
            line = f.readline().strip()
            while line:
                month, year, published_as = line.strip().split(' ', 2)
                flag = published_as.find('PrepTest')
                if flag > 0:
                    _, test_num = published_as.rsplit(' ', 1)
                    test_num = int(test_num[:-1])
                else:
                    test_num = 0
                for game_num in range(1, 5): # index games from 1 to 4
                    _, _, game_attr = f.readline().strip().split(' ', 2)
                    game_attr = game_attr.split(', ') # yields a list of varying length
                    game_type = game_attr[0].split(': ')
                    if len(game_type) == 1:
                        game_type += [''] + [''] # fill missing entries
                    elif len(game_type) == 2:
                        game_type += [''] # fill missing entries
                    game_id = [month, year, published_as[1:-1], test_num, game_num]
                            # drop parentheses around publication info
                    own_col = ['missing'] # for ownership fill with default value
                    all_cols = game_id + game_type + own_col + game_attr[1:]
                    array.append(all_cols)
                line = f.readline().strip()
        return array

    def populate_lists(self):
        '''
        Generate list versions of sentences from prompts and rules
        '''
        for game in Lsat.keyed.iterrows():
            output = []
            for sent in prompts[game[0]]:
                output.append([w for w in sent.split(' ')])

            output = []
            for sent in rules[game[0]]:
                output.append([w for w in sent.split(' ')])
        return output

    def _editor(self, line):
        '''
        Add space to prevent SpaCy from mistaking some variables for abreviations.
        '''
        if len(line) < 2:
            return line
        elif (line[-1]=='.'):
            if line[-2].isupper():
                line = line[:-1] + ' .'
            if line[-2].isdigit():
                line = line[:-1] + ' .'
        return line

    def read_prompts(self):
        names = ['Book_One_prompts.txt', 'Vol_V_prompts.txt', \
                'New_Actual.txt', '10_Actual.txt', 'Book_Two.txt']
        for name in names:
            type = 'error'
            # print (name)
            with open('../data/' + name) as f:
                line = f.readline().strip()
                while line != '--END--':
                    #print (line)
                    if line[:4]=="####": # new type
                        contents = line.split(' ')
                        _, *type, _ = contents
                        line = f.readline().strip()
                    if line[:3]=="###": # new game
                        contents = line.split(' ')
                        # print ('lenth of line 2', len(contents))
                        _, month, year, _, game_num, _ = contents
                        #line = f.readline().strip()
                        # print
                        print (name, *type, 'mo', month, 'yr', year, 'game', game_num)
                        line = self._editor(f.readline().strip())
                        prompt = []
                        while (line[:2]!='##') and (line[:2]!='**'):
                            prompt.append(line)
                            line = self._editor(f.readline().strip())
                        rule_list = []
                        line = self._editor(f.readline().strip())
                        while (line[:2]!='##') and (line!='--END--') and (line[:2]!='**'):
                            rule_list.append(line)
                            line = self._editor(f.readline().strip())
                            #print ("rule line", line)
                        quest_list = []
                        quest_labels = []
                        all_five_answers_list = [] # without letter
                        all_five_answers_list_raw = [] # with letter attached
                        print ('questions to parse?')
                        print ('current line is ', line)
                        while line[:2]=='**': # check if have any questions to parse
                            # for now can ignore line containing original question numbers
                            question = f.readline().strip() # get question
                            q_labels = f.readline().strip().split(',') # get comma-sep labels
                            quest_list.append(question) # save questions
                            quest_labels.append(q_labels) # save labels
                            line = f.readline().strip() # pull up next line

                            answer_list = []
                            answer_list_raw = []
                            print ('answers to parse?')
                            while line[:1]=='%': # check if have any answers to parse
                                print ('current answer is ', line)
                                # for now can ignore line containing original question number
                                answer_list.append(line[6:]) # save answers without letters
                                answer_list_raw.append(line[2:]) # save answers with letters
                                line = f.readline().strip() # pull up next line
                            all_five_answers_list.append(answer_list)
                            all_five_answers_list_raw.append(answer_list_raw)

                        subset = df[df.year==year]
                        subset = subset[subset.month==month]
                        subset = subset[subset.game_num==int(game_num)]          #(df.game_num==int(game_num))]
                        #print (month, year, int(game_num))
                        try:
                            idx = subset.index.tolist()[0]
                            prompts[idx] = prompt
                            rules[idx] = rule_list
                            questions[idx] = quest_list
                            question_labels[idx] = quest_labels
                            answers[idx] = all_five_answers_list
                            answers_raw[idx] = all_five_answers_list_raw
                        except:
                            print ("error: missing", month, year, int(game_num))

    def get_owned(self, df):
        df.loc[df.test_num.isin(range(1, 21)), 'own'] = 'Book One'
        df.loc[df.test_num.isin(range(21, 41)), 'own'] = 'Book Two' # missing PrepTest 41
        df.loc[df.test_num.isin(range(42, 52)), 'own'] = '10 Actual'
        df.loc[df.test_num.isin(range(52, 62)), 'own'] = '10 New Actual'
        df.loc[df.test_num.isin(range(62, 72)), 'own'] = 'Vol V'
        df.loc[(df.month == 'June') & (df.year == '2007'), 'own'] = 'Free'
        return df[df.own != 'missing']

    def get_counts(self, input, col_names):
        type_counts = input.groupby('primary_type').count().secondary_type.to_frame(col_names[0])
        type_counts[col_names[1]] = type_counts[col_names[0]] * 100/ len(input)
        return type_counts

def save_pickle(file, name):
    with open("../data/" + name + ".pkl", 'wb') as f:
        pickle.dump(file, f)
    print ("done pickling ", name)

def load_pickle(name):
    '''
    Return unpickled file
    '''
    with open("classification_data/" + name + ".pkl") as f_un:
        file_unpickled = pickle.load(f_un)
    print ("done unpickling ", name)
    return file_unpickled

def flatten_question_labels():
    array = [label for labels in question_labels for label in labels]
    return array

if __name__ == '__main__':
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.width', 300)

    Lsat = LSAT()
    file = '../data/Games_Classifications.txt'
    cols = ['month', 'year', 'published_as', 'test_num', 'game_num', 'primary_type', \
        'secondary_type', 'tertiary_type', 'own', 'notes1', 'notes2', 'notes3']
    array = Lsat.load(file)

    df = pd.DataFrame(array, columns=cols)
    df.index = df.index + 1 # since game_num starts at 1, test_num should too
    # for convenience create various subsets of the df:
    df_owned = Lsat.get_owned(df) # games I own ~ training data
    bible_games = df[df.values == 'Bible'] # games dissected in PowerScore's Bible
    tests_inventory = df.groupby('own').count() # inventory of where owned games reside
    total_type_counts = Lsat.get_counts(df, ['total_counts', 'percent_overall'])
    owned_type_counts = Lsat.get_counts(df_owned, ['owned_counts', 'percent_of_owned'])
    combined_type_counts = pd.concat([total_type_counts,owned_type_counts], axis=1)
    combined_type_counts['held_out'] = combined_type_counts['total_counts']- \
                                       combined_type_counts['owned_counts']
    combined_type_counts['percent_held'] = combined_type_counts['held_out'] * 100/ \
                                   combined_type_counts['total_counts']

    all_seq_games = df[df.primary_type=='Pure Sequencing']
    seq_games_owned = all_seq_games[all_seq_games.own != 'missing']

    prompts = [[] for i in range(df.shape[0])]
    rules = [[] for i in range(df.shape[0])]
    questions = [[] for i in range(df.shape[0])]
    question_labels = [[] for i in range(df.shape[0])]
    answers = {} # key = (game_num, quest_num, answer_num)
    prompts_as_spdoc = [[] for i in range(df.shape[0])] # each sent sep in list
    rules_as_spdoc = [[] for i in range(df.shape[0])] # each sent sep in list
    questions_as_spdoc = [[] for i in range(df.shape[0])] # each sent sep in list
    answers_as_spdoc = {} # key = (game_num, quest_num, answer_num) # extra nested list
    prompts_1_spdoc = [[] for i in range(df.shape[0])] # all prompts as 1 SpaCy doc
    rules_1_spdoc = [[] for i in range(df.shape[0])] # all rules as 1 SpaCy doc
    questions_1_spdoc = [[] for i in range(df.shape[0])] # all questions as 1 SpaCy doc

    Lsat.read_prompts()
    df.game_num[9]=2 # manual correction
    df.game_num[35]=1 # manual correction
    df.primary_type[9]='Basic Linear' # manual correction
    df.primary_type[35]='Basic Linear' # manual correction
    counter = 0
    df['keyed_pr'] = False
    for i, p in enumerate(prompts):
        if len(p)>0:
            counter += 1
            # print (counter, df.index[i], df.year[i], df.month[i], df.primary_type[i], df.own[i])
            df.keyed_pr[i] = True
    Lsat.keyed = df[df.keyed_pr] # subset = those with prompts and rules keyed in
    Lsat.keyed_seq = Lsat.keyed[Lsat.keyed.primary_type=='Pure Sequencing']
    Lsat.keyed_seq_lin = Lsat.keyed[Lsat.keyed.primary_type.isin(['Basic Linear', 'Pure Sequencing'])]
    ql_array = flatten_question_labels()
    ql_df = pd.DataFrame(ql_array, columns=['local', 'type', 'misc1', 'misc2'])

    print ("Total sequencing games:", len(all_seq_games))
    print ("Owned sequencing games:", len(seq_games_owned))
    print ("Keyed sequencing games:", len(Lsat.keyed_seq))
    print (Lsat.keyed.groupby('primary_type').count())

    Lsat.populate_lists()
    Lsat.df = df
    Lsat.prompts = prompts
    Lsat.rules = rules
    Lsat.questions = questions
    Lsat.question_labels = question_labels
    Lsat.answers = answers #
    Lsat.prompts_as_spdoc = prompts_as_spdoc
    Lsat.rules_as_spdoc = rules_as_spdoc
    Lsat.questions_as_spdoc = questions_as_spdoc
    Lsat.answers_as_spdoc = answers_as_spdoc # dict
    Lsat.prompts_1_spdoc = prompts_1_spdoc
    Lsat.rules_1_spdoc = rules_1_spdoc
    Lsat.questions_1_spdoc = questions_1_spdoc

    save_pickle(Lsat, 'LSAT_data')

    '''
                                 total_counts  percent_overall  owned_counts  percent_of_owned  held_out  percent_held
Advanced Linear                        85        23.876404          71.0         25.000000      14.0     16.470588
Basic Linear                           79        22.191011          67.0         23.591549      12.0     15.189873
Circular Linearity                      5         1.404494           2.0          0.704225       3.0     60.000000
Grouping                              122        34.269663          97.0         34.154930      25.0     20.491803
Grouping/Linear Combination            21         5.898876          16.0          5.633803       5.0     23.809524
Linear/Grouping Combination             2         0.561798           NaN               NaN       NaN           NaN
Linear/Mapping Combination              1         0.280899           NaN               NaN       NaN           NaN
Mapping                                 1         0.280899           1.0          0.352113       0.0      0.000000
Mapping-Spatial Relations               2         0.561798           2.0          0.704225       0.0      0.000000
Mapping-Supplied Diagram                3         0.842697           3.0          1.056338       0.0      0.000000
Pattern                                12         3.370787           8.0          2.816901       4.0     33.333333
Pure Sequencing                        23         6.460674          17.0          5.985915       6.0     26.086957

'''
