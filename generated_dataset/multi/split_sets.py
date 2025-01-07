import json
with open('generated_dataset/multi/objects_dict.json') as f:
    metadata = json.load(f)
lst = list(metadata.values())
import random
random.shuffle(lst)
dct = {}
for i in range(len(lst)):
    if i  < 4:
        dct[lst[i]] = 'images'
    elif i < 8:
        dct[lst[i]] = 'right_videos'
    elif i < 12:
        dct[lst[i]] = 'left_videos'
    elif i < 16:
        dct[lst[i]] = 'images_right_videos'
    elif i < 20:
        dct[lst[i]] = 'images_left_videos'
    elif i < 24:
        dct[lst[i]] = 'right_videos_left_videos'
    elif i < 28:
        dct[lst[i]] = 'images_right_videos_left_videos'
    else:
        dct[lst[i]] = 'none'
with open('generated_dataset/multi/split_sets.json', 'w') as f:
    json.dump(dct, f)