import json
import pandas as pd

val_num = 300
base_num = 4000
    
split_sets_names = ['static', 'dynamic', 'both']

METADATA_PATH = 'generated_dataset/multiple/metadata.csv'
METADATA_PATH_TARGET = 'generated_dataset/multiple/metadata_splited.csv'
data = pd.read_csv(METADATA_PATH)
# add a new column to the dataframe, called usage_aug, and set all values to 'tbc'

for i, split_sets_name in enumerate(split_sets_names):
    # select rows with object name
    rows = data[(data['split_name'] == split_sets_name) & (data['usage'] == 'tbc')]
    select_rows = rows.sample(val_num)
    data.loc[select_rows.index, 'usage'] = 'val_image'
    
    rows = data[(data['split_name'] == split_sets_name) & (data['usage'] == 'tbc')]
    select_rows = rows.sample(val_num)
    data.loc[select_rows.index, 'usage'] = 'val_video' 
    

    if split_sets_name == 'dynamic':
        rows = data[(data['split_name'] == split_sets_name) & (data['usage'] == 'tbc')]
        select_rows = rows.sample(base_num)
        data.loc[select_rows.index, 'usage'] = 'train_video'
        
    elif split_sets_name == 'static':
        rows = data[(data['split_name'] == split_sets_name) & (data['usage'] == 'tbc')]
        select_rows = rows.sample(base_num)
        data.loc[select_rows.index, 'usage'] = 'train_image'
        
    elif split_sets_name == 'both':
        rows = data[(data['split_name'] == split_sets_name) & (data['usage'] == 'tbc')]
        select_rows = rows.sample(base_num)
        data.loc[select_rows.index, 'usage'] = 'train_image'
        
        rows = data[(data['split_name'] == split_sets_name) & (data['usage'] == 'tbc')]
        select_rows = rows.sample(base_num)
        data.loc[select_rows.index, 'usage'] = 'train_video'
    else:
        pass

rows = data[(data['split_name'] == split_sets_name) & (data['usage'] == 'train_video')]
print(len(rows))

data.to_csv(METADATA_PATH_TARGET, index=False)
