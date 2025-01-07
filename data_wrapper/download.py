import os

import boto3
from tqdm import tqdm

from data_wrapper.utils import load_json, dump_json, s3_download_file, s3_file_exists

DATA_DIR = 'proc_data/data/ActivityNet'
RAW_VIDEO_DIR = os.path.join(DATA_DIR, 'videos')
os.makedirs(RAW_VIDEO_DIR, exist_ok=True)
S3_AGENT = boto3.client("s3")

# Load the ActivityNet captions
TRAIN_CAPS = load_json(os.path.join(DATA_DIR, 'train.json'))
VAL_CAPS = load_json(os.path.join(DATA_DIR, 'val_1.json'))
# TEST_CAPS = load_json(DATA_DIR, 'val_2.json')  # we won't use this
# as `val_1.json` is a superset of `val_2.json`


def download_video(video_fn, caps=None):
    if not s3_file_exists(
        bucket='snap-small-video-datasets',
        folder='ActivityNet/videos/',
        filename=video_fn,
        s3=S3_AGENT,
    ):
        print(f"Video {video_fn} not found in S3. Skipping...")
        return False
    s3_download_file(
        bucket='snap-small-video-datasets',
        folder='ActivityNet/videos/',
        filename=video_fn,
        local_folder=RAW_VIDEO_DIR,
        s3=S3_AGENT
    )
    if caps is not None:
        caps = {
            f'event_{i}': {
                'start_t': caps['timestamps'][i][0],
                'end_t': caps['timestamps'][i][1],
                'caption': caps['sentences'][i],
            } for i in range(len(caps['sentences']))
        }
        dump_json(caps, os.path.join(RAW_VIDEO_DIR, video_fn.replace('.mp4', '.summary_text.json')), indent=2)
    return True


if __name__ == '__main__':
    video_keys = list(TRAIN_CAPS.keys())
    cnt = 0
    for k in video_keys:
        video_fn = k[2:] if k.startswith('v_') else k
        success = download_video(video_fn + '.mp4', caps=TRAIN_CAPS[k])
        if success:
            cnt += 1
        if cnt >= 10:
            break
