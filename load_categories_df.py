import pandas as pd

def load(file):
    ''' loads game categorizations and returns a list of lists:
    [game_id, game_type]
    game_id = (month, year, publication_information, game_number)
    game_type = type, [additional info, additional info]
    '''

    array = []
    with open(file) as f:
        line = f.readline().strip()
        while line:
            month, year, published_as = line.strip().split(' ', 2)
            for game_number in range(1, 5):
                _, _, game_type = f.readline().strip().split(' ', 2)
                game_type = game_type.split(', ') # yields a list of varying length
                game_id = [month, year, published_as[1:-1], game_number]
                        # drop parentheses around publication info
                all_cols = game_id + game_type
                array.append(all_cols)
            line = f.readline().strip()
    return array

if __name__ == '__main__':
    file = 'Games_Classifications.txt'
    cols = ['month', 'year', 'published_as', 'game#', 'primary_type', \
    'secondary_type', 'tertiary_type', 'misc1', 'misc2', 'misc3']
    array = load(file)
    df = pd.DataFrame(array, columns=cols)
    print (df.head(1))
