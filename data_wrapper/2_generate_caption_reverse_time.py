import pandas as pd
import json
import random

DATA_CAPTION_DIR = '/fsx/yban/myproject/kubric/generated_dataset/reverse_time/output_caption_part0.csv'
RATIO = 0.9
df = pd.read_csv(DATA_CAPTION_DIR)
paths =df.path.tolist()
captions = df.text.tolist()

lst = list(range(len(paths)))
random.shuffle(lst)
spilt = int(len(lst) * RATIO)
training_set = lst[:spilt]
val_set = lst[spilt:]

training_captions_dict = {}
val_captions_dict = {}

for i in training_set:
    path = paths[i]
    caption = captions[i]
    video_name = path.split('/')[-1].replace(".mp4", "")
    cap_key = "v_" + video_name
    training_captions_dict[cap_key] = caption
    
    
for i in val_set:
    path = paths[i]
    caption = captions[i]
    video_name = path.split('/')[-1].replace(".mp4", "")
    cap_key = "v_" + video_name
    val_captions_dict[cap_key] = caption
    
with open('/fsx/yban/myproject/kubric/generated_dataset/reverse_time/train.json', 'w') as f:
    json.dump(training_captions_dict, f)

with open('/fsx/yban/myproject/kubric/generated_dataset/reverse_time/val.json', 'w') as f:
    json.dump(val_captions_dict, f)