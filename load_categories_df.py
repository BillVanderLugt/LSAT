import pandas as pd

def load(file):
    '''
    loads game categorizations and returns a array
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

def read_prompts():
    names = ['../Book_One_prompts.txt', '../Vol_V_prompts.txt', '../New_Actual.txt']
    for name in names:
        type = 'error'
        print (name)
        with open(name) as f:
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
                    print (name, *type, 'mo', month, 'yr', year, 'game', game_num)
                    line = f.readline().strip()
                    prompt = []
                    while line[:2]!='##':
                        prompt.append(line)
                        line = f.readline().strip()
                        print ("prompt line", line)
                    #line = f.readline() # eat spacer
                    rule_list = []
                    line = f.readline().strip()
                    while line[:2]!='##' and line!='--END--':
                        rule_list.append(line)
                        line = f.readline().strip()
                        print ("rule line", line)
                    # print (name, *type, 'mo', month, 'yr', year, 'game', game_num)
                    # print ("PROMPT:")
                    # print (prompt)
                    # print ("rules:")
                    # print (rule_list)
                    # print (month, year, int(game_num))
                    subset = df[df.year==year]
                    subset = subset[subset.month==month]
                    subset = subset[subset.game_num==int(game_num)]          #(df.game_num==int(game_num))]
                    print (month, year, int(game_num))
                    try:
                        idx = subset.index.tolist()[0]
                        prompts[idx] = prompt
                        rules[idx] = rule_list
                    except:
                        print ("error: missing", month, year, int(game_num))

def get_owned(df):
    df.loc[df.test_num.isin(range(1, 21)), 'own'] = 'Book One'
    df.loc[df.test_num.isin(range(21, 41)), 'own'] = 'Book Two' # missing PrepTest 41
    df.loc[df.test_num.isin(range(42, 52)), 'own'] = '10 Actual'
    df.loc[df.test_num.isin(range(52, 62)), 'own'] = '10 New Actual'
    df.loc[df.test_num.isin(range(62, 72)), 'own'] = 'Vol V'
    df.loc[(df.month == 'June') & (df.year == '2007'), 'own'] = 'Free'
    return df[df.own != 'missing']

def get_counts(input, col_names):
    type_counts = input.groupby('primary_type').count().secondary_type.to_frame(col_names[0])
    type_counts[col_names[1]] = type_counts[col_names[0]] * 100/ len(input)
    return type_counts

def main():
    pass

if __name__ == '__main__':
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.width', 300)

    file = 'Games_Classifications.txt'
    cols = ['month', 'year', 'published_as', 'test_num', 'game_num', 'primary_type', \
        'secondary_type', 'tertiary_type', 'own', 'notes1', 'notes2', 'notes3']
    array = load(file)

    df = pd.DataFrame(array, columns=cols)
    df.index = df.index + 1 # since game_num starts at 1, test_num should too

    # for convenience create various subsets of the df:
    df_owned = get_owned(df) # games I own ~ training data
    bible_games = df[df.values == 'Bible'] # games dissected in PowerScore's Bible
    tests_inventory = df.groupby('own').count() # inventory of where owned games reside
    total_type_counts = get_counts(df, ['total_counts', 'percent_overall'])
    owned_type_counts = get_counts(df_owned, ['owned_counts', 'percent_of_owned'])
    combined_type_counts = pd.concat([total_type_counts,owned_type_counts], axis=1)
    combined_type_counts['held_out'] = combined_type_counts['total_counts']- \
                                       combined_type_counts['owned_counts']
    combined_type_counts['percent_held'] = combined_type_counts['held_out'] * 100/ \
                                   combined_type_counts['total_counts']
    #print (combined_type_counts)
    prompts = [[] for i in range(df.shape[0])]
    rules = [[] for i in range(df.shape[0])]
    #print ('init as', prompts)
    read_prompts()
    df.game_num[9]=2 # manual correction of wierd error
    df.game_num[35]=1 # manual correction of wierd error
    df.primary_type[9]='Basic Linear' # manual correction of wierd error
    df.primary_type[35]='Basic Linear' # manual correction of wierd error
    counter = 0
    df['keyed_pr'] = False
    for i, p in enumerate(prompts):
        if len(p)>0:
            counter += 1
            print (counter, df.year[i], df.month[i], df.primary_type[i], df.own[i])
            df.keyed_pr[i] = True
    keyed = df[df.keyed_pr] # subset = those with prompts and rules keyed in
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

1 1991 June Pure Sequencing Book One
2 1991 October Pure Sequencing Book One
3 1991 October Basic Linear Book One
4 1991 December Advanced Linear Book One
5 1991 December Basic Linear Book One
6 1991 December Grouping Book One
7 1992 February Pure Sequencing Book One
8 1992 February Grouping Book One
9 1992 June Basic Linear Book One
10 1992 June Grouping Book One
11 1992 June Grouping Book One
12 1992 October Pure Sequencing Book One
13 1992 December Grouping Book One
14 1992 December Grouping Book One
15 1993 February Basic Linear Book One
16 1993 June Basic Linear Book One
17 1993 June Advanced Linear Book One
18 1993 October Grouping Book One
19 1994 February Pure Sequencing Book One
20 1994 June Grouping Book One
21 1994 June Grouping Book One
22 1994 October Grouping Book One
23 1994 October Grouping Book One
24 1994 December Grouping Book One
25 1994 December Basic Linear Book One
26 1995 February Grouping Book One
27 1995 June Basic Linear Book One
28 1995 June Grouping Book One
29 1995 September Grouping Book One
30 1995 December Basic Linear Book One
31 1995 December Grouping Book One
32 1996 June Basic Linear Book One
33 1996 June Grouping Book One
34 1996 June Grouping Book One
35 2007 September Pure Sequencing 10 New Actual
36 2007 December Pure Sequencing 10 New Actual
37 2008 October Pure Sequencing 10 New Actual
38 2010 October Pure Sequencing 10 New Actual
39 2010 December Basic Linear Vol V
40 2010 December Basic Linear Vol V
41 2011 June Basic Linear Vol V
42 2011 June Basic Linear Vol V
43 2011 June Basic Linear Vol V
44 2011 October Basic Linear Vol V
45 2011 December Basic Linear Vol V
46 2011 December Basic Linear Vol V
47 2012 June Basic Linear Vol V
48 2012 October Basic Linear Vol V
49 2013 June Basic Linear Vol V
50 2013 October Basic Linear Vol V
51 2013 December Pure Sequencing Vol V
52 2013 December Basic Linear Vol V

                 month  year  published_as  test_num  game_num  secondary_type  tertiary_type  own  notes1  notes2  notes3  keyed_pr
primary_type
Basic Linear        24    24            24        24        24              24             24   24       8       0       0        24
Grouping            18    18            18        18        18              18             18   18      15       3       0        18
Pure Sequencing     10    10            10        10        10              10             10   10       1       0       0        10
'''
