import numpy as np
import pandas as pd

data_dir = '/Users/sherifabdelkarim/projects/lifelong_fact_learning/data/mid_scale/splits/'
split = '4tasks_random/'
data_dir += split
total_facts = []
for b in range(1, 5):
    b_test = pd.read_csv(data_dir + 'B{}_test.csv'.format(b))
    facts = pd.read_csv(data_dir + 'B{}_facts.csv'.format(b))
    total_facts.append(facts)
    facts = facts.set_index('id')
    gt = facts.loc[b_test.id, ['type', 'S', 'P', 'O']]
    gt = gt.reset_index()
    # gt.to_csv('gt_words/' + split + 'B{}_gt.csv'.format(b), index=False)

total_facts = pd.concat(total_facts, ignore_index=True)

print(len(total_facts))
print(len(total_facts.drop_duplicates()))

# print(pd.unique(total_facts['S']))
# print(pd.unique(total_facts['P']))
# print(pd.unique(total_facts['O']))
#
# print(total_facts['S'].drop_duplicates().dropna().to_string())
# print(total_facts['P'].drop_duplicates().dropna().to_string())
# print(total_facts['O'].drop_duplicates().dropna().to_string())

# total_facts['SO'] = pd.concat([total_facts['S'], total_facts['O']])
SO = list(total_facts['S']) + list(total_facts['O'])
SO = pd.Series(SO)
SO = SO.drop_duplicates().dropna()
# print(total_facts['SO'])
# print(total_facts['SO'])
total_facts['S'].drop_duplicates().dropna().to_csv('gt_words/' + split + 'S.csv', index=False)
total_facts['P'].drop_duplicates().dropna().to_csv('gt_words/' + split + 'P.csv', index=False)
total_facts['O'].drop_duplicates().dropna().to_csv('gt_words/' + split + 'O.csv', index=False)
SO.to_csv('gt_words/' + split + 'SO.csv', index=False)
