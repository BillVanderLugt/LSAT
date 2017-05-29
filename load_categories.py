import pandas

def load(file):
    ''' loads game categorizations and returns a list of lists:
    [game_id, game_type]
    game_id = (month, year, publication_information, game_number)
    game_type = type, [additional info, additional info]
    '''
    cols = ['#', 'month', 'year', 'published_as', 'game#', 'type', 'misc1', 'misc2']
    df = pd.DataFrame(columns=cols)
    classifications = [
    with open(file) as f:
        line = f.readline().strip()
        counter = 1
        while line:
            month, year, published_as = line.strip().split(' ', 2)
            for game_number in range(1, 5):
                _, _, game_type = f.readline().strip().split(' ', 2)
                game_type = game_type.split(', ') # yields a list of varying length
                game_id = month, year, published_as[1:-1], game_number
                        # drop parentheses around publication info
                classifications.append([game_id, *game_type])
            counter += 1
            line = f.readline().strip()
    return classifications

if __name__ == '__main__':
    file = 'Games_Classifications.txt'
    classifications = load(file)
