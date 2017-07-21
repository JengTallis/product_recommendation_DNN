import pandas as pd
df = pd.read_csv('num.csv')
df = df.sort_values(['cusId', 'fetchDate'], ascending = [True, True])
df.to_csv('Full_List_sorted.csv', index=False, sep='\t', encoding='utf-8')