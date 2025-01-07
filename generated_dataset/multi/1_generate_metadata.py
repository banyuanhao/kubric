import os
import json
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import random

# Load asset and background mappings
with open('generated_dataset/reverse_time_video_and_image/asset_id_to_name.json') as f:
    asset_id_to_name = json.load(f)
with open('generated_dataset/reverse_time_video_and_image/background_id_to_name.json') as f:
    background_id_to_name = json.load(f)
with open('generated_dataset/multi/split_sets.json') as f:
    split_sets = json.load(f)

BASE_PATH = '/nfs/yban/dataset/multi'
META_PATH = 'generated_dataset/multi'
METADATA_PATH = f'{BASE_PATH}/rawdata'
VIDEO_PATH = f'{BASE_PATH}/videos'

metadata_names = os.listdir(METADATA_PATH)
metadata_names = [name for name in metadata_names if name.endswith('.json') and 'metadata' in name]

def process_metadata(metadata_name):
    with open(os.path.join(METADATA_PATH, metadata_name)) as f:
        metadata = json.load(f)

    if metadata['metadata']['num_instances'] == 0:
        return None
    id =metadata_name.split('.')[0].split('_')[1]
    object_name = asset_id_to_name[metadata['instances'][0]['asset_id']]
    background = background_id_to_name[metadata['metadata']['background']]
    direction = metadata['direction']
    caption_video = f"{object_name} is moving to the {direction} in the {background}"
    caption_image = f"{object_name} is in the {background}"
    split_set = split_sets[object_name]
    # random select 1 int froom 8 to 16 
    image_selecting = random.randint(8, 16)
    image_path = os.path.join(METADATA_PATH, f'{id}/rgba_{image_selecting:05d}.png')

    return {
        'path': os.path.join(VIDEO_PATH, metadata_name.replace('metadata', 'video').replace('.json', '.mp4')),
        'text_video': caption_video,
        'text_image': caption_image,
        'num_frames': 24,
        'object': object_name,
        'background': background,
        'direction': direction,
        'split_set': split_set,
        'id': id,
        'image_path': image_path,
        'image_selecting': image_selecting,
        'usage': 'tbc',
        'usage_bal': 'tbc'
    }

# Use multiprocessing to process metadata files in parallel
with Pool(cpu_count()) as pool:
    results = list(tqdm(pool.imap(process_metadata, metadata_names), total=len(metadata_names)))

# Filter out None results
results = [result for result in results if result is not None]

# Convert the results into a DataFrame and save it
data = pd.DataFrame(results)
data.to_csv(f'{META_PATH}/metadata_aug.csv', index=False)
