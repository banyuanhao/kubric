import pandas as pd
import random
import os
BASE_PATH = '/fsx_ori/yban/dataset/multiple_64f'
META_PATH = 'generated_dataset/multiple_64f'
METADATA_PATH = f'{BASE_PATH}/rawdata'

data = pd.read_csv(META_PATH + '/metadata_splited.csv')

ids = data.id.tolist()
image_paths = []
selected_integers = []
for id in ids:
    selected_integer = random.sample(range(0, 64), 8)
    image_path = [os.path.join(METADATA_PATH, f'{id:05d}/rgba_{image_selecting:05d}.png') for image_selecting in selected_integer]
    image_paths.append(image_path)
    selected_integers.append(selected_integer)
data['image_path'] = image_paths
data['image_selecting'] = selected_integers
data.to_csv(META_PATH + '/metadata_splited_.csv', index=False)

