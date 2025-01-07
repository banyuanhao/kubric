import pandas as pd
import os
import json
with open('generated_dataset/multi/split_sets.json') as f:
    split_sets = json.load(f)

data = pd.read_csv('generated_dataset/multi/metadata_aug_splited.csv')
# csv2 = pd.read_csv('generated_dataset/multi/metadata_aug.csv')
# print(csv2.shape)
# # merge csv1 and csv2, csv1 is a subset of csv2. in the first rows, it should be csv1 in the same order, and then csv2 without the rows that are in csv1
# csv = pd.concat([csv1, csv2[~csv2.id.isin(csv1.id)]])
# csv.to_csv('generated_dataset/multi/metadata_aug_splited.csv', index=False)

rows = data[(data['object'].map(split_sets) == 'images_left_videos') & (data['usage_bal'] == 'train_image')]
print(len(rows))