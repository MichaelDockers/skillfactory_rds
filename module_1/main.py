import pandas as pd
from itertools import chain, combinations
from collections import Counter


def most_frequent(list_in):
    return Counter(list_in).most_common(1)


def gen_answer_key():
    counter = 0
    while True:
        counter += 1
        yield f'{counter:02}'


def dict_add(lst_in):
    answers[next(it)] = lst_in


answers = {}
data = pd.read_csv('movie_bd_v5.csv')

df = data.copy()
it = gen_answer_key()

df['profit'] = df['revenue'] - df['budget']
df['title_len'] = df['original_title'].apply(lambda x: len(str(x)))
df['overview_counter'] = df['overview'].apply(lambda x: len(x.split()))
df['director'] = df['director'].str.split('|')
df['genres'] = df['genres'].str.split('|')
df['cast'] = df['cast'].str.split('|')
df['production_companies'] = df['production_companies'].str.split('|')
df['release_date'] = pd.to_datetime(df['release_date'], format='%m/%d/%Y')
output_list = ['imdb_id', 'profit', 'original_title', 'release_year']

print(df.info())
print(df.columns)

print()
print('1. The most budget movie:')
print(df[df['budget'] == df['budget'].max()][['imdb_id', 'budget', 'original_title']].to_string(header=False, index=False))

print()
print('2. The longest movie:')
print(df[df['runtime'] == df['runtime'].max()][['imdb_id', 'runtime', 'original_title']])

print()
print('3. The shortest movie:')
print(df[df['runtime'] == df['runtime'].min()][['imdb_id', 'runtime', 'original_title']])

print()
print('4. Mean runtime movie:')
print(round(df['runtime'].mean()))

print()
print('5. Median runtime movie:')
print(df['runtime'].median())

print()
print('6. The most profit movie:')
print(df[df['profit'] == df['profit'].max()][output_list])

print()
print('7. Worst profit movie:')
print(df[df['profit'] == df['profit'].min()][output_list])

print()
print('8. Number of profit movies:')
print(df[df['profit'] > 0]['profit'].count())

print()
print('9. Most revenue 2008 year movie:')
print(df[df['release_year'] == 2008][['imdb_id', 'revenue', 'original_title']].nlargest(1, ['revenue']))

print()
print('10. Best loser 2012 - 2014 year movie:')
print(df[(df['release_year'] >= 2012) & (df['release_year'] <= 2014)][output_list].nsmallest(1, ['profit']))

print()
print('11. The most common genre:')
print(most_frequent(list(df.explode('genres')['genres'])))
print(most_frequent(list(chain.from_iterable(df['genres']))))

print()
print('12. The most common genre in profit movies:')
print(most_frequent(list(chain.from_iterable(df[df.profit > 0]['genres']))))
print(most_frequent(list(df[df['profit'] > 0].explode('genres')['genres'])))

print()
print('13. The most revenue director:')
print(df.explode('director').groupby('director')['revenue'].sum().idxmax())

print()
print('14. The most action director:')
print(df[df.genres.str.join('').str.contains('Action')].explode('director').
      groupby('director')['genres'].count().idxmax())

print()
print('15. The actor with biggest sum revenue in 2012 year:')
print(df[df['release_year'] == 2012].explode('cast').groupby('cast')['revenue'].sum().idxmax())

print()
print('16. The most budget actor (more than mean):')
print(df[df['budget'] > df['budget'].mean()].explode('cast').groupby('cast')['cast'].count().idxmax())

print()
print('17. Most common genre of Nicolas Cage:')
print(df[df.cast.str.join('').str.contains('Nicolas Cage')].explode('genres').
      groupby('genres')['genres'].count().idxmax())

print()
print('18. The worth movie of Paramount Pictures:')
print(df[df.production_companies.str.join('').str.contains('Paramount Pictures')]
      [output_list].nsmallest(1, 'profit'))

print()
print('19. The best year of sum revenue:')
print(df.groupby('release_year')['revenue'].sum().idxmax())

print()
print('20. Most profit year of Warner Bros?:')
print(df[df.production_companies.str.join('').str.contains('Warner Bros')].groupby('release_year')
      ['profit'].sum().idxmax())

print()
print('21. The most productive month for all years:')
print(df.groupby(df['release_date'].dt.month)['imdb_id'].count().idxmax())

print()
print('22. Sum of summer movies for all years:')
print(df[df['release_date'].dt.month.isin([6, 7, 8])]['imdb_id'].count())

print()
print('23. The most winter director:')
print(df[df['release_date'].dt.month.isin([1, 2, 12])].
      explode('director').groupby('director')['imdb_id'].count().idxmax())

print()
print('24. Longest title studio:')
print(df.explode('production_companies').groupby('production_companies')['title_len'].max().idxmax())

print()
print('25. The longest mean (by words) overview studio:')
print(df.explode('production_companies').groupby('production_companies')['overview_counter'].mean().idxmax())

print()
print('26. 1 percent best rating movies:')
print(df[df['vote_average'] >= df['vote_average'].quantile(0.99)]
      [['original_title', 'vote_average']].nlargest(5, 'vote_average'))

print()
print('23. The most winter director:')
df['cast'] = df['cast'].apply(lambda x: list(combinations(x, 2)))
print(df.explode('cast')['cast'].value_counts().nlargest(1))

for key, item in answers.items():
    print(key, item)
