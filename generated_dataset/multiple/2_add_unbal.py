import json
import pandas as pd

val_num = 300
base_num = 4000
    
split_sets_names = ['static', 'dynamic', 'both']

METADATA_PATH = 'generated_dataset/multiple/metadata.csv'
METADATA_PATH_TARGET = 'generated_dataset/multiple/metadata_splited.csv'
data = pd.read_csv(METADATA_PATH_TARGET)
# add two new column to the dataframe, called usage_bal, and usage_unbal, and set values equal to usage
data['usage_bal'] = data['usage']
data['usage_unbal'] = data['usage']

for i, split_sets_name in enumerate(split_sets_names):
    
    if split_sets_name == 'both':
        rows = data[(data['split_name'] == split_sets_name) & (data['usage'] == 'train_video')]
        # select half of the rows
        select_rows = rows.sample(len(rows)//2)
        data.loc[select_rows.index, 'usage_bal'] = 'tbc'
        data.loc[select_rows.index, 'usage_unbal'] = 'tbc'
        
        rows = data[(data['split_name'] == split_sets_name) & (data['usage'] == 'train_image')]
        select_rows = rows.sample(len(rows)//2)
        data.loc[select_rows.index, 'usage_bal'] = 'tbc'
        data.loc[select_rows.index, 'usage_unbal'] = 'tbc'
        
    elif split_sets_name == 'static':
        rows = data[(data['split_name'] == split_sets_name) & (data['usage'] == 'train_image')]
        select_rows = rows.sample(len(rows)//2)
        data.loc[select_rows.index, 'usage_bal'] = 'train_video'
        
    elif split_sets_name == 'dynamic':
        rows = data[(data['split_name'] == split_sets_name) & (data['usage'] == 'train_video')]
        select_rows = rows.sample(len(rows)//2)
        data.loc[select_rows.index, 'usage_bal'] = 'train_image'
    else:
        pass

data.to_csv(METADATA_PATH_TARGET, index=False)

rows = data[(data['split_name'] == 'both') & (data['usage_bal'] == 'train_video')]
print(len(rows))