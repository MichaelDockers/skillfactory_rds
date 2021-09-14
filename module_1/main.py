import pandas as pd
from itertools import chain, combinations
from collections import Counter


def most_frequent(list_in):
    """Function to count strings in list. Output one most common string (can be changed to another value if needed)"""
    return Counter(list_in).most_common(1)


def gen_answer_key():
    """Generator of endless nums (output string)"""
    counter = 0
    while True:
        counter += 1
        yield f'{counter:02}'


def dict_add(lst_in):
    """Func for adding string to dict with auto generated keys"""
    answers[next(it)] = lst_in


def print_dict():
    """Func for print answers dict combining with questions"""
    for quest, answer in zip(questions, answers.items()):
        print(f'{answer[0]}. - {quest} {answer[1]}')


# Creating dict to answers
answers = {}
questions = [
    'The most budget movie:',
    'The longest movie:',
    'The shortest movie:',
    'Mean runtime movie:',
    'Median runtime movie:',
    'The most profit movie:',
    'Worst profit movie:',
    'Number of profit movies:',
    'Most revenue 2008 year movie:',
    'Best loser 2012 - 2014 year movie:',
    'The most common genre:',
    'The most common genre in profit movies:',
    'The most revenue director:',
    'The most action director:',
    'The actor with biggest sum revenue in 2012 year:',
    'The most budget actor (more than mean):',
    'Most common genre of Nicolas Cage:',
    'The worth movie of Paramount Pictures:',
    'The best year of sum revenue:',
    'Most profit year of Warner Bros?:',
    'The most productive month for all years:',
    'Sum of summer movies for all years:',
    'The most winter director:',
    'Longest title studio:',
    'The longest mean (by words) overview studio:',
    '1 percent best rating movies:',
    'The most common duet',
]

# Reading data from CSV file
data = pd.read_csv('movie_bd_v5.csv')

# Creating copy of original dataframe to manipulate
df = data.copy()

# Creating iterator fot dict keys
it = gen_answer_key()

# Making some processing on copy of dataframe
df['profit'] = df['revenue'] - df['budget']
df['title_len'] = df['original_title'].apply(lambda x: len(str(x)))
df['overview_counter'] = df['overview'].apply(lambda x: len(x.split()))
df['director'] = df['director'].str.split('|')
df['genres'] = df['genres'].str.split('|')
df['cast'] = df['cast'].str.split('|')
df['cast_treat'] = df['cast'].apply(lambda x: list(combinations(x, 2)))
df['production_companies'] = df['production_companies'].str.split('|')
df['release_date'] = pd.to_datetime(df['release_date'], format='%m/%d/%Y')
output_list = ['imdb_id', 'profit', 'original_title', 'release_year']

print(df.info())
print(df.columns)

# Quest 1
dict_add(df[df['budget'] == df['budget'].max()]['original_title'].to_string(header=False, index=False))

# Quest 2
dict_add(df[df['runtime'] == df['runtime'].max()]['original_title'].to_string(header=False, index=False))

# Quest 3
dict_add(df[df['runtime'] == df['runtime'].min()]['original_title'].to_string(header=False, index=False))

# Quest 4
dict_add(round(df['runtime'].mean()))

# Quest 5
dict_add(int(df['runtime'].median()))

# Quest 6
dict_add(df[df['profit'] == df['profit'].max()]['original_title'].to_string(header=False, index=False))

# Quest 7
dict_add(df[df['profit'] == df['profit'].min()]['original_title'].to_string(header=False, index=False))

# Quest 8
dict_add(df[df['profit'] > 0]['profit'].count())

# Quest 9
dict_add(df[df['release_year'] == 2008][['original_title', 'revenue']].
         nlargest(1, ['revenue']).to_string(header=False, index=False))

# Quest 10
dict_add(df[(df['release_year'] >= 2012) & (df['release_year'] <= 2014)]
         [['original_title', 'profit']].nsmallest(1, ['profit']).to_string(header=False, index=False))

# Quest 11 (option 1)
dict_add(most_frequent(list(df.explode('genres')['genres'])))

# Quest 11 (option 2)
# dict_add(most_frequent(list(chain.from_iterable(df['genres']))))

# Quest 12 (option 1)
dict_add(most_frequent(list(chain.from_iterable(df[df.profit > 0]['genres']))))

# Quest 12 (option 2)
# dict_add(most_frequent(list(df[df['profit'] > 0].explode('genres')['genres'])))

# Quest 13
dict_add(df.explode('director').groupby('director')['revenue'].sum().idxmax())

# Quest 14
dict_add(df[df.genres.str.join('').str.contains('Action')].explode('director').
         groupby('director')['genres'].count().idxmax())

# Quest 15
dict_add(df[df['release_year'] == 2012].explode('cast').groupby('cast')['revenue'].sum().idxmax())

# Quest 16
dict_add(df[df['budget'] > df['budget'].mean()].explode('cast').groupby('cast')['cast'].count().idxmax())

# Quest 17
dict_add(df[df.cast.str.join('').str.contains('Nicolas Cage')].explode('genres').
         groupby('genres')['genres'].count().idxmax())

# Quest 18
dict_add(df[df.production_companies.str.join('').str.contains('Paramount Pictures')]
         [['original_title', 'profit']].nsmallest(1, 'profit').to_string(header=False, index=False))

# Quest 19
dict_add(df.groupby('release_year')['revenue'].sum().idxmax())

# Quest 20
dict_add(df[df.production_companies.str.join('').str.contains('Warner Bros')].groupby('release_year')
         ['profit'].sum().idxmax())

# Quest 21
dict_add(df.groupby(df['release_date'].dt.month)['imdb_id'].count().idxmax())

# Quest 22
dict_add(df[df['release_date'].dt.month.isin([6, 7, 8])]['imdb_id'].count())

# Quest 23
dict_add(df[df['release_date'].dt.month.isin([1, 2, 12])].
         explode('director').groupby('director')['imdb_id'].count().idxmax())

# Quest 24
dict_add(df.explode('production_companies').groupby('production_companies')['title_len'].max().idxmax())

# Quest 25
dict_add(df.explode('production_companies').groupby('production_companies')['overview_counter'].mean().idxmax())

# Quest 26
dict_add(df[df['vote_average'] >= df['vote_average'].quantile(0.99)]
         [['original_title', 'vote_average']].nlargest(5, 'vote_average').to_string(header=False, index=False))

# Quest 27
dict_add(df.explode('cast_treat')['cast_treat'].value_counts().nlargest(1).to_string(header=False))

print_dict()
