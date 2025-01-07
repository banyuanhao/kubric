import os
import json
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import random

# Load asset and background mappings
with open('generated_dataset/multiple/asset_id_to_name.json') as f:
    asset_id_to_name = json.load(f)
with open('generated_dataset/multiple/background_id_to_name.json') as f:
    background_id_to_name = json.load(f)

BASE_PATH = '/nfs/yban/dataset/multiple'
META_PATH = 'generated_dataset/multiple'
METADATA_PATH = f'{BASE_PATH}/rawdata'
VIDEO_PATH = f'{BASE_PATH}/videos'

metadata_names = os.listdir(METADATA_PATH)
metadata_names = [name for name in metadata_names if name.endswith('.json') and 'metadata' in name]

def process_metadata(metadata_name):
    with open(os.path.join(METADATA_PATH, metadata_name)) as f:
        metadata = json.load(f)

    if metadata['metadata']['num_instances'] != 1:
        return None
    
    velocities = metadata['velocities']
    split_name = metadata['split_name']
    
    id =metadata_name.split('.')[0].split('_')[1]
    
    object_name = asset_id_to_name[metadata['instances'][0]['asset_id']]
    background = background_id_to_name[metadata['metadata']['background']]

    caption_video = f"{object_name} is dropping in the {background}"
    caption_image = f"{object_name} in the {background}"

    # random select 1 int froom 8 to 16 
    image_selecting = random.randint(0, 16)
    image_path = os.path.join(METADATA_PATH, f'{id}/rgba_{image_selecting:05d}.png')

    return {
        'path': os.path.join(VIDEO_PATH, metadata_name.replace('metadata', 'video').replace('.json', '.mp4')),
        'text_video': caption_video,
        'text_image': caption_image,
        'num_frames': 24,
        'object': object_name,
        'velcoities': velocities,
        'background': background,
        'split_name': split_name,
        'id': id,
        'image_path': image_path,
        'image_selecting': image_selecting,
        'usage': 'tbc',
    }

# Use multiprocessing to process metadata files in parallel
with Pool(cpu_count()) as pool:
    results = list(tqdm(pool.imap(process_metadata, metadata_names), total=len(metadata_names)))

# Filter out None results
results = [result for result in results if result is not None]

# Convert the results into a DataFrame and save it
data = pd.DataFrame(results)
data.to_csv(f'{META_PATH}/metadata.csv', index=False)
