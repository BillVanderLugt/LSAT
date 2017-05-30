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
            for game_num in range(1, 5):
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

def get_owned(df):
    df.loc[df.test_num.isin(range(1, 21)), 'own'] = 'Book One'
    df.loc[df.test_num.isin(range(21, 41)), 'own'] = 'Book Two' # missing PrepTest 41
    df.loc[df.test_num.isin(range(42, 52)), 'own'] = '10 Actual'
    df.loc[df.test_num.isin(range(52, 62)), 'own'] = 'Vol V'
    df.loc[df.test_num.isin(range(62, 72)), 'own'] = '10 New Actual'
    df.loc[(df.month == 'June') & (df.year == '2007'), 'own'] = 'Free'
    return df[df.own != 'missing']

def get_counts(input, col_names):
    type_counts = input.groupby('primary_type').count().secondary_type.to_frame(col_names[0])
    type_counts[col_names[1]] = type_counts[col_names[0]] * 100/ len(input)
    return type_counts


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
    owned_type_counts = get_counts(df_owned, ['owned_counts', 'percent of owned'])
    combined_type_counts = pd.concat([total_type_counts,owned_type_counts], axis=1)
    combined_type_counts['held_out'] = combined_type_counts['total_counts']- \
                                       combined_type_counts['owned_counts']
    combined_type_counts['percent_held'] = combined_type_counts['held_out'] * 100/ \
                                   combined_type_counts['total_counts']
    print (combined_type_counts)
