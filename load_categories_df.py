import pandas as pd
import pickle

class LSAT(object):

    def __init__(self):
        self.prompts = []
        self.rules = []
        self.prompts_pos = []
        self.rules_pos = []

    def load(self, file):
        '''
        loads game categorizations and returns a array,
        ready for conversion into a pandas DataFrame
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
        generate list versions of sentences from prompts and rules
        '''
        for game in Lsat.keyed.iterrows():
            # print ('############## processing game #: {} #################'.format(game[0]))
            output = []
            for sent in prompts[game[0]]:
                # print (sent)
                output.append([w for w in sent.split(' ')])
            prompts_as_list[game[0]] = output

            output = []
            for sent in rules[game[0]]:
                # print (sent)
                output.append([w for w in sent.split(' ')])
            rules_as_list[game[0]] = output
        return output

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
                    if line[:4]=="####":
                        contents = line.split(' ')
                        _, *type, _ = contents
                        line = f.readline().strip()
                    if line[:3]=="###":
                        contents = line.split(' ')
                        # print ('lenth of line 2', len(contents))
                        _, month, year, _, game_num, _ = contents
                        #line = f.readline().strip()
                        # print
                        # print (name, *type, 'mo', month, 'yr', year, 'game', game_num)
                        line = f.readline().strip()
                        prompt = []
                        while line[:2]!='##':
                            prompt.append(line)
                            line = f.readline().strip()
                            #print ("prompt line", line)
                        #line = f.readline() # eat spacer
                        rule_list = []
                        line = f.readline().strip()
                        while line[:2]!='##' and line!='--END--':
                            rule_list.append(line)
                            line = f.readline().strip()
                            #print ("rule line", line)
                        # print (name, *type, 'mo', month, 'yr', year, 'game', game_num)
                        # print ("PROMPT:")
                        # print (prompt)
                        # print ("rules:")
                        # print (rule_list)
                        # print (month, year, int(game_num))
                        subset = df[df.year==year]
                        subset = subset[subset.month==month]
                        subset = subset[subset.game_num==int(game_num)]          #(df.game_num==int(game_num))]
                        #print (month, year, int(game_num))
                        try:
                            idx = subset.index.tolist()[0]
                            prompts[idx] = prompt
                            rules[idx] = rule_list
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
    returns unpickled file
    '''
    with open("classification_data/" + name + ".pkl") as f_un:
        file_unpickled = pickle.load(f_un)
    print ("done unpickling ", name)
    return file_unpickled

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
    #df['indices'] = df.index
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
    prompts_cleaned = [[] for i in range(df.shape[0])]
    rules_cleaned = [[] for i in range(df.shape[0])]
    prompts_as_list = [[] for i in range(df.shape[0])]
    rules_as_list = [[] for i in range(df.shape[0])]
    prompts_pos_as_list = [[] for i in range(df.shape[0])]
    rules_pos_as_list = [[] for i in range(df.shape[0])]
    prompts_pos_plus_punct = [[] for i in range(df.shape[0])]
    rules_pos_plus_punct = [[] for i in range(df.shape[0])]
    #print ('init as', prompts)
    Lsat.read_prompts()
    df.game_num[9]=2 # manual correction of wierd error
    df.game_num[35]=1 # manual correction of wierd error
    df.primary_type[9]='Basic Linear' # manual correction of wierd error
    df.primary_type[35]='Basic Linear' # manual correction of wierd error
    counter = 0
    df['keyed_pr'] = False
    for i, p in enumerate(prompts):
        if len(p)>0:
            counter += 1
            print (counter, df.index[i], df.year[i], df.month[i], df.primary_type[i], df.own[i])
            df.keyed_pr[i] = True
    Lsat.keyed = df[df.keyed_pr] # subset = those with prompts and rules keyed in
    Lsat.keyed_seq = Lsat.keyed[Lsat.keyed.primary_type=='Pure Sequencing']

    #print (df.groupby('primary_type').count())
    print ("Total sequencing games:", len(all_seq_games))
    print ("Owned sequencing games:", len(seq_games_owned))
    #print (keyed_seq)
    print ("Keyed sequencing games:", len(Lsat.keyed_seq))

    Lsat.populate_lists()
    Lsat.df = df
    Lsat.prompts = prompts
    Lsat.rules = rules
    Lsat.prompts_cleaned = prompts_cleaned
    Lsat.rules_cleaned = rules_cleaned
    Lsat.prompts_as_list = prompts_as_list
    Lsat.rules_as_list = rules_as_list
    Lsat.prompts_pos_as_list = prompts_pos_as_list
    Lsat.rules_pos_as_list = rules_pos_as_list
    Lsat.prompts_pos_plus_punct = prompts_pos_plus_punct
    Lsat.rules_pos_plus_punct = rules_pos_plus_punct
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


2 6 1991 October Pure Sequencing Book One
3 7 1991 October Basic Linear Book One
4 10 1991 December Basic Linear Book One
5 11 1991 December Basic Linear Book One
6 13 1991 December Grouping Book One
7 14 1992 February Pure Sequencing Book One
8 15 1992 February Grouping Book One
9 18 1992 June Basic Linear Book One
10 19 1992 June Grouping Book One
11 20 1992 June Grouping Book One
12 23 1992 October Pure Sequencing Book One
13 26 1992 December Grouping Book One
14 29 1992 December Grouping Book One
15 30 1993 February Basic Linear Book One
16 34 1993 June Basic Linear Book One
17 36 1993 June Basic Linear Book One
18 39 1993 October Grouping Book One
19 42 1994 February Pure Sequencing Book One
20 46 1994 June Grouping Book One
21 48 1994 June Grouping Book One
22 51 1994 October Grouping Book One
23 52 1994 October Grouping Book One
24 54 1994 December Grouping Book One
25 55 1994 December Basic Linear Book One
26 58 1995 February Grouping Book One
27 62 1995 June Basic Linear Book One
28 65 1995 June Grouping Book One
29 66 1995 September Grouping Book One
30 70 1995 December Basic Linear Book One
31 71 1995 December Grouping Book One
32 78 1996 June Basic Linear Book One
33 80 1996 June Grouping Book One
34 81 1996 June Grouping Book One
35 187 2004 June Pure Sequencing 10 Actual
36 207 2005 December Pure Sequencing 10 Actual
37 217 2006 September Pure Sequencing 10 Actual
38 221 2006 December Pure Sequencing 10 Actual
39 229 2007 September Pure Sequencing 10 New Actual
40 231 2007 December Pure Sequencing 10 New Actual
41 240 2008 October Pure Sequencing 10 New Actual
42 263 2010 October Pure Sequencing 10 New Actual
43 266 2010 December Basic Linear Vol V
44 269 2010 December Basic Linear Vol V
45 271 2011 June Basic Linear Vol V
46 272 2011 June Basic Linear Vol V
47 273 2011 June Basic Linear Vol V
48 274 2011 October Basic Linear Vol V
49 278 2011 December Basic Linear Vol V
50 281 2011 December Basic Linear Vol V
51 283 2012 June Basic Linear Vol V
52 287 2012 October Basic Linear Vol V
53 294 2013 June Basic Linear Vol V
54 298 2013 October Basic Linear Vol V
55 302 2013 December Pure Sequencing Vol V
56 305 2013 December Basic Linear Vol V


KEYED              month  year  published_as  test_num  game_num  secondary_type  tertiary_type  own  notes1  notes2  notes3  keyed_pr
primary_type
Basic Linear        24    24            24        24        24              24             24   24       8       0       0        24
Grouping            18    18            18        18        18              18             18   18      15       3       0        18
Pure Sequencing     14    14            14        14        14              14             14   14       2       0       0        14
'''
