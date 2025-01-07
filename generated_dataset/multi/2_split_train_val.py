import json
import pandas as pd

val_num = 120
base_num = 30

with open('generated_dataset/multi/objects_dict.json') as f:
    metadata = json.load(f)
    
with open('generated_dataset/multi/split_sets.json') as f:
    split_sets = json.load(f)
    
objects = list(metadata.values())

METADATA_PATH = 'generated_dataset/multi/metadata_aug_splited.csv'
METADATA_PATH_TARGET = 'generated_dataset/multi/metadata_aug_splited_.csv'
data = pd.read_csv(METADATA_PATH)
# add a new column to the dataframe, called usage_aug, and set all values to 'tbc'
data['usage_aug'] = 'tbc'

for i, object in enumerate(objects):
    # select rows with object name
    rows = data[(data['object'] == object) & (data['usage_aug'] == 'tbc')]
    select_rows = rows.sample(val_num)
    data.loc[select_rows.index, 'usage_aug'] = 'val_image'
    
    # select rows with object name and direction right
    rows = data[(data['object'] == object) & (data['direction'] == 'right') & (data['usage_aug'] == 'tbc')]
    select_rows = rows.sample(val_num)
    data.loc[select_rows.index, 'usage_aug'] = 'val_video'
    
    # select rows with object name and direction left
    rows = data[(data['object'] == object) & (data['direction'] == 'left') & (data['usage_aug'] == 'tbc')]
    select_rows = rows.sample(val_num)
    data.loc[select_rows.index, 'usage_aug'] = 'val_video'   
    

    if split_sets[object] == 'images':
        rows = data[(data['object'] == object) & (data['usage_aug'] == 'tbc')]
        select_rows = rows.sample(base_num*6)
        data.loc[select_rows.index, 'usage_aug'] = 'train_image'
        
    elif split_sets[object] == 'right_videos':
        rows = data[(data['object'] == object) & (data['direction'] == 'right') & (data['usage_aug'] == 'tbc')]
        select_rows = rows.sample(base_num*6)
        data.loc[select_rows.index, 'usage_aug'] = 'train_video'
        
    elif split_sets[object] == 'left_videos':
        rows = data[(data['object'] == object) & (data['direction'] == 'left') & (data['usage_aug'] == 'tbc')]
        select_rows = rows.sample(base_num*6)
        data.loc[select_rows.index, 'usage_aug'] = 'train_video'
        
    elif split_sets[object] == 'images_right_videos':
        rows = data[(data['object'] == object) & (data['usage_aug'] == 'tbc')]
        select_rows = rows.sample(base_num*6)
        data.loc[select_rows.index, 'usage_aug'] = 'train_image'
        rows = data[(data['object'] == object) & (data['direction'] == 'right') & (data['usage_aug'] == 'tbc')]
        select_rows = rows.sample(base_num*6)
        data.loc[select_rows.index, 'usage_aug'] = 'train_video'
        
    elif split_sets[object] == 'images_left_videos':
        rows = data[(data['object'] == object) & (data['usage_aug'] == 'tbc')]
        select_rows = rows.sample(base_num*6)
        data.loc[select_rows.index, 'usage_aug'] = 'train_image'
        rows = data[(data['object'] == object) & (data['direction'] == 'left') & (data['usage_aug'] == 'tbc')]
        select_rows = rows.sample(base_num*6)
        data.loc[select_rows.index, 'usage_aug'] = 'train_video'
    
    elif split_sets[object] == 'right_videos_left_videos':
        rows = data[(data['object'] == object) & (data['direction'] == 'right') & (data['usage_aug'] == 'tbc')]
        select_rows = rows.sample(base_num*3)
        data.loc[select_rows.index, 'usage_aug'] = 'train_video'
        rows = data[(data['object'] == object) & (data['direction'] == 'left') & (data['usage_aug'] == 'tbc')]
        select_rows = rows.sample(base_num*3)
        data.loc[select_rows.index, 'usage_aug'] = 'train_video'
        
    elif split_sets[object] == 'images_right_videos_left_videos':
        rows = data[(data['object'] == object) & (data['usage_aug'] == 'tbc')]
        select_rows = rows.sample(base_num*6)
        data.loc[select_rows.index, 'usage_aug'] = 'train_image'
        rows = data[(data['object'] == object) & (data['direction'] == 'right') & (data['usage_aug'] == 'tbc')]
        select_rows = rows.sample(base_num*3)
        data.loc[select_rows.index, 'usage_aug'] = 'train_video'
        rows = data[(data['object'] == object) & (data['direction'] == 'left') & (data['usage_aug'] == 'tbc')]
        select_rows = rows.sample(base_num*3)
        data.loc[select_rows.index, 'usage_aug'] = 'train_video'
        
        
    else:
        pass

rows = data[(data['object'].map(split_sets) == 'right_videos') & (data['usage_aug'] == 'train_video')]
print(len(rows))
rows = data[(data['object'].map(split_sets) == 'images_right_videos') & (data['usage_aug'] == 'train_video')]
print(len(rows))
rows = data[(data['object'].map(split_sets) == 'images_right_videos_left_videos') & (data['usage_aug'] == 'val_video')]
print(len(rows))

data.to_csv(METADATA_PATH_TARGET, index=False)
