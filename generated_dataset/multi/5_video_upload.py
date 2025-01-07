from datetime import datetime
from pytz import timezone
import multiprocessing
import os
import boto3
import argparse
import random
from tqdm import tqdm
import pandas as pd
from utils import compress_into_tar, load_json

DATA_DIR = '/fsx/yban/dataset/multi'
VIDEO_DIR = os.path.join(DATA_DIR, 'videos_our_format')
IMAGE_DIR = os.path.join(DATA_DIR, 'images_our_format')
SHARD_DIR = os.path.join(DATA_DIR, 'video_shards_our_format')
os.makedirs(SHARD_DIR, exist_ok=True)

DATA_CAPTION_DIR = 'generated_dataset/multi/metadata_splited.csv'

def create_webdataset_image(input_args):
    video_filenames, node_id, task_id, shard_size_mb, output_log_file = input_args

    shard_size = 0
    shard_id = 0
    shard_filename = f"n{node_id:05d}_t{task_id:05d}_s{shard_id:07d}.tar"
    upload_list = []

    s3 = boto3.client("s3")
    for video_filename in tqdm(video_filenames):
        # Collect files to download
        filenames = [
            [video_filename, True],
            [video_filename.replace(".png", ".summary_text.json"), True],
            [video_filename.replace(".png", ".summary_text_embeddings.pkl"), True],
            [video_filename.replace(".png", ".imginfo.json"), True],
        ]
        cur_video = os.path.join(IMAGE_DIR, video_filename)
        cur_vidinfo = cur_video.replace(".png", ".imginfo.json")
        if not os.path.exists(cur_video) or not os.path.exists(cur_vidinfo):
            print(f"Video {cur_video} not found. Skipping...")
            continue

        # Register files to be included in a shard
        upload_list_per_video = [os.path.join(IMAGE_DIR, i[0]) for i in filenames]
        if all([os.path.isfile(i) for i in upload_list_per_video]) is False:
            print([os.path.isfile(i) for i in upload_list_per_video])
            print()
        assert all([os.path.isfile(i) for i in upload_list_per_video]) 
        
        upload_list += [i for i in upload_list_per_video if os.path.isfile(i)]

        # Gets the size of the current video
        video_size = os.path.getsize(os.path.join(IMAGE_DIR, video_filename))
        video_size_mb = video_size / (1024**2)
        shard_size += video_size_mb

        if shard_size > shard_size_mb:
            # Create shard
            compress_into_tar(upload_list, os.path.join(SHARD_DIR, shard_filename))

            # Uploads the shard to S3
            s3.upload_file(os.path.join(SHARD_DIR, shard_filename), "snap-webdataset-videos", "multi/" + shard_filename)
            with open(output_log_file, "a") as f:
                video_count = sum(1 for video_file in upload_list if video_file.endswith(".png"))
                f.write(datetime.now(timezone("US/Pacific")).strftime("%m/%d %H:%M:%S") + " %s (including %i videos) is uploaded\n"%(shard_filename, video_count))

            # Initialize a new shard and increment shard_id
            shard_size = 0
            shard_id += 1
            shard_filename = f"n{node_id:05d}_t{task_id:05d}_s{shard_id:07d}.tar"
            upload_list = []

    # Flush the last shard
    if shard_size > 0:
        # Create shard
        compress_into_tar(upload_list, os.path.join(SHARD_DIR, shard_filename))

        # Uploads the shard to S3
        s3.upload_file(os.path.join(SHARD_DIR, shard_filename), "snap-webdataset-videos", "multi/" + shard_filename)
        with open(output_log_file, "a") as f:
            video_count = sum(1 for video_file in upload_list if video_file.endswith(".png"))
            f.write(datetime.now(timezone("US/Pacific")).strftime("%m/%d %H:%M:%S") + " %s (including %i videos) is uploaded\n"%(shard_filename, video_count))


def create_webdataset(input_args):
    video_filenames, node_id, task_id, shard_size_mb, output_log_file = input_args

    shard_size = 0
    shard_id = 0
    shard_filename = f"n{node_id:05d}_t{task_id:05d}_s{shard_id:07d}.tar"
    upload_list = []

    s3 = boto3.client("s3")
    for video_filename in tqdm(video_filenames):
        # Collect files to download
        filenames = [
            [video_filename, True],
            [video_filename.replace(".mp4", ".summary_text.json"), True],
            [video_filename.replace(".mp4", ".summary_text_embeddings.pkl"), True],
            [video_filename.replace(".mp4", ".vidinfo.json"), True],
        ]
        cur_video = os.path.join(VIDEO_DIR, video_filename)
        cur_vidinfo = cur_video.replace(".mp4", ".vidinfo.json")
        if not os.path.exists(cur_video) or not os.path.exists(cur_vidinfo):
            print(f"Video {cur_video} not found. Skipping...")
            continue

        # Register files to be included in a shard
        upload_list_per_video = [os.path.join(VIDEO_DIR, i[0]) for i in filenames]
        if not all([os.path.isfile(i) for i in upload_list_per_video]):
            print(f"Video {cur_video} is missing some files. Skipping...")
            break
        upload_list += [i for i in upload_list_per_video if os.path.isfile(i)]

        # Gets the size of the current video
        video_size = os.path.getsize(os.path.join(VIDEO_DIR, video_filename))
        video_size_mb = video_size / (1024**2)
        shard_size += video_size_mb

        if shard_size > shard_size_mb:
            # Create shard
            compress_into_tar(upload_list, os.path.join(SHARD_DIR, shard_filename))

            # Uploads the shard to S3
            s3.upload_file(os.path.join(SHARD_DIR, shard_filename), "snap-webdataset-videos", "multi/" + shard_filename)
            with open(output_log_file, "a") as f:
                video_count = sum(1 for video_file in upload_list if video_file.endswith(".mp4"))
                f.write(datetime.now(timezone("US/Pacific")).strftime("%m/%d %H:%M:%S") + " %s (including %i videos) is uploaded\n"%(shard_filename, video_count))

            # Initialize a new shard and increment shard_id
            shard_size = 0
            shard_id += 1
            shard_filename = f"n{node_id:05d}_t{task_id:05d}_s{shard_id:07d}.tar"
            upload_list = []

    # Flush the last shard
    if shard_size > 0:
        # Create shard
        compress_into_tar(upload_list, os.path.join(SHARD_DIR, shard_filename))

        # Uploads the shard to S3
        s3.upload_file(os.path.join(SHARD_DIR, shard_filename), "snap-webdataset-videos", "multi/" + shard_filename)
        with open(output_log_file, "a") as f:
            video_count = sum(1 for video_file in upload_list if video_file.endswith(".mp4"))
            f.write(datetime.now(timezone("US/Pacific")).strftime("%m/%d %H:%M:%S") + " %s (including %i videos) is uploaded\n"%(shard_filename, video_count))


if __name__ == "__main__":
    # python upload_activitynet_webdataset.py --split train
    # python upload_activitynet_webdataset.py --split val
    parser = argparse.ArgumentParser(description="Create webdataset")
    # parser.add_argument("--split", type=str, required=True, choices=["train", "val",'images','right_videos','left_videos','images_right_videos','images_left_videos','images_right_videos_left_videos','right_videos_left_videos'])
    parser.add_argument("--node-id", type=int, default=0)
    parser.add_argument("--num-node", type=int, default=1)
    parser.add_argument("--shard-size-mb", type=int, default=1000)
    args = parser.parse_args()

    metadata = pd.read_csv(DATA_CAPTION_DIR)
    
    lst = ['train_video','train_image','val_video','val_image', 'images__left','images__right','right_videos__right','right_videos__left','left_videos__left','left_videos__right','images_right_videos__left','images_right_videos__right','images_left_videos__left','images_left_videos__right','images_right_videos_left_videos__left','images_right_videos_left_videos__right','right_videos_left_videos__left','right_videos_left_videos__right','train_video_bal','train_image_bal','train_right_videos__right','train_left_videos__left','train_images_right_videos__right','train_images_left_videos__left']
    # lst = ['train_video_bal','train_image_bal']
    
    for split in lst:
    
        if split == "train_video":
            video_list = metadata[(metadata['usage'] == 'train_video')]
            video_list = video_list.id.tolist()
            video_list = ["multi_video_%s.mp4" % str(vn).zfill(5) for vn in video_list]
            node_id = args.node_id
        elif split == "train_image":
            video_list = metadata[(metadata['usage'] == 'train_image')]
            video_list = video_list.id.tolist()
            video_list = ["multi_image_%s.png" % str(vn).zfill(5) for vn in video_list]
            node_id = args.node_id + 100
        elif split == 'val_video':
            video_list = metadata[(metadata['usage'] == 'val_video')]
            video_list = video_list.id.tolist()
            video_list = ["multi_video_%s.mp4" % str(vn).zfill(5) for vn in video_list]
            node_id = args.node_id + 200
        elif split == 'val_image':
            video_list = metadata[(metadata['usage'] == 'val_image')]
            video_list = video_list.id.tolist()
            video_list = ["multi_image_%s.png" % str(vn).zfill(5) for vn in video_list]
            node_id = args.node_id + 300
        elif split == 'images__left':
            video_list = metadata[(metadata['split_set'] == 'images') & (metadata['direction'] == 'left') & (metadata['usage'] == 'val_video')]
            video_list = video_list.id.tolist()
            video_list = ["multi_video_%s.mp4" % str(vn).zfill(5) for vn in video_list]
            node_id = args.node_id + 400
        elif split == 'images__right':
            video_list = metadata[(metadata['split_set'] == 'images') & (metadata['direction'] == 'right') & (metadata['usage'] == 'val_video')]
            video_list = video_list.id.tolist()
            video_list = ["multi_video_%s.mp4" % str(vn).zfill(5) for vn in video_list]
            node_id = args.node_id + 500
        elif split == 'right_videos__right':
            video_list = metadata[(metadata['split_set'] == 'right_videos') & (metadata['direction'] == 'right') & (metadata['usage'] == 'val_video')]
            video_list = video_list.id.tolist()
            video_list = ["multi_video_%s.mp4" % str(vn).zfill(5) for vn in video_list]
            node_id = args.node_id + 600
        elif split == 'right_videos__left':
            video_list = metadata[(metadata['split_set'] == 'right_videos') & (metadata['direction'] == 'left') & (metadata['usage'] == 'val_video')]
            video_list = video_list.id.tolist()
            video_list = ["multi_video_%s.mp4" % str(vn).zfill(5) for vn in video_list]
            node_id = args.node_id + 700
        elif split == 'left_videos__left':
            video_list = metadata[(metadata['split_set'] == 'left_videos') & (metadata['direction'] == 'left') & (metadata['usage'] == 'val_video')]
            video_list = video_list.id.tolist()
            video_list = ["multi_video_%s.mp4" % str(vn).zfill(5) for vn in video_list]
            node_id = args.node_id + 800
        elif split == 'left_videos__right':
            video_list = metadata[(metadata['split_set'] == 'left_videos') & (metadata['direction'] == 'right') & (metadata['usage'] == 'val_video')]
            video_list = video_list.id.tolist()
            video_list = ["multi_video_%s.mp4" % str(vn).zfill(5) for vn in video_list]
            node_id = args.node_id + 900
        elif split == 'images_right_videos__left':
            video_list = metadata[(metadata['split_set'] == 'images_right_videos') & (metadata['direction'] == 'left') & (metadata['usage'] == 'val_video')]
            video_list = video_list.id.tolist()
            video_list = ["multi_video_%s.mp4" % str(vn).zfill(5) for vn in video_list]
            node_id = args.node_id + 1000
        elif split == 'images_right_videos__right':
            video_list = metadata[(metadata['split_set'] == 'images_right_videos') & (metadata['direction'] == 'right') & (metadata['usage'] == 'val_video')]
            video_list = video_list.id.tolist()
            video_list = ["multi_video_%s.mp4" % str(vn).zfill(5) for vn in video_list]
            node_id = args.node_id + 1100
        elif split == 'images_left_videos__left':
            video_list = metadata[(metadata['split_set'] == 'images_left_videos') & (metadata['direction'] == 'left') & (metadata['usage'] == 'val_video')]
            video_list = video_list.id.tolist()
            video_list = ["multi_video_%s.mp4" % str(vn).zfill(5) for vn in video_list]
            node_id = args.node_id + 1200
        elif split == 'images_left_videos__right':
            video_list = metadata[(metadata['split_set'] == 'images_left_videos') & (metadata['direction'] == 'right') & (metadata['usage'] == 'val_video')]
            video_list = video_list.id.tolist()
            video_list = ["multi_video_%s.mp4" % str(vn).zfill(5) for vn in video_list]
            node_id = args.node_id + 1300
        elif split == 'images_right_videos_left_videos__left':
            video_list = metadata[(metadata['split_set'] == 'images_right_videos_left_videos') & (metadata['direction'] == 'left') & (metadata['usage'] == 'val_video')]
            video_list = video_list.id.tolist()
            video_list = ["multi_video_%s.mp4" % str(vn).zfill(5) for vn in video_list]
            node_id = args.node_id + 1400
        elif split == 'images_right_videos_left_videos__right':
            video_list = metadata[(metadata['split_set'] == 'images_right_videos_left_videos') & (metadata['direction'] == 'right') & (metadata['usage'] == 'val_video')]
            video_list = video_list.id.tolist()
            video_list = ["multi_video_%s.mp4" % str(vn).zfill(5) for vn in video_list]
            node_id = args.node_id + 1500
        elif split == 'right_videos_left_videos__left':
            video_list = metadata[(metadata['split_set'] == 'right_videos_left_videos') & (metadata['direction'] == 'left') & (metadata['usage'] == 'val_video')]
            video_list = video_list.id.tolist()
            video_list = ["multi_video_%s.mp4" % str(vn).zfill(5) for vn in video_list]
            node_id = args.node_id + 1600
        elif split == 'right_videos_left_videos__right':
            video_list = metadata[(metadata['split_set'] == 'right_videos_left_videos') & (metadata['direction'] == 'right') & (metadata['usage'] == 'val_video')]
            video_list = video_list.id.tolist()
            video_list = ["multi_video_%s.mp4" % str(vn).zfill(5) for vn in video_list]
            node_id = args.node_id + 1700
        elif split == 'train_video_bal':
            video_list = metadata[(metadata['usage_bal'] == 'train_video')]
            video_list = video_list.id.tolist()
            video_list = ["multi_video_%s.mp4" % str(vn).zfill(5) for vn in video_list]
            node_id = args.node_id + 1800
        elif split == 'train_image_bal':
            video_list = metadata[(metadata['usage_bal'] == 'train_image')]
            video_list = video_list.id.tolist()
            video_list = ["multi_image_%s.png" % str(vn).zfill(5) for vn in video_list]
            node_id = args.node_id + 1900
        elif split == 'train_right_videos__right':
            video_list = metadata[(metadata['split_set'] == 'right_videos') & (metadata['direction'] == 'right') & (metadata['usage'] == 'train_video')]
            video_list = video_list.id.tolist()
            video_list = ["multi_video_%s.mp4"  % str(vn).zfill(5) for vn in video_list]
            node_id = args.node_id + 2000
        elif split == 'train_left_videos__left':
            video_list = metadata[(metadata['split_set'] == 'left_videos') & (metadata['direction'] == 'left') & (metadata['usage'] == 'train_video')]
            video_list = video_list.id.tolist()
            video_list = ["multi_video_%s.mp4"  % str(vn).zfill(5) for vn in video_list]
            node_id = args.node_id + 2100
        elif split == 'train_images_right_videos__right':
            video_list = metadata[(metadata['split_set'] == 'images_right_videos') & (metadata['direction'] == 'right') & (metadata['usage'] == 'train_video')]
            video_list = video_list.id.tolist()
            video_list = ["multi_video_%s.mp4"  % str(vn).zfill(5) for vn in video_list]
            node_id = args.node_id + 2200
        elif split == 'train_images_left_videos__left':
            video_list = metadata[(metadata['split_set'] == 'images_left_videos') & (metadata['direction'] == 'left') & (metadata['usage'] == 'train_video')]
            video_list = video_list.id.tolist()
            video_list = ["multi_video_%s.mp4"  % str(vn).zfill(5) for vn in video_list]
            node_id = args.node_id + 2300
            
            
        else:
            raise NotImplementedError
        
        output_log_file = os.path.join(SHARD_DIR, f"{split}_logs.txt")

        # random shuffle subjects
        random.seed(0)
        random.shuffle(video_list)

        # split for each node
        video_list = video_list[args.node_id::args.num_node]

        # split for each process
        num_parallel_process = 10
        video_list = [video_list[i::num_parallel_process] for i in range(num_parallel_process)]

        input_args = [
            (
                video_list[task_id],
                node_id,
                task_id,
                args.shard_size_mb,
                output_log_file
            )
            for task_id in range(num_parallel_process)
        ]
        
        if split == 'val_image' or split == 'train_image' or split =='train_image_bal':
            with multiprocessing.Pool(num_parallel_process) as p:
                _ = p.map(create_webdataset_image, input_args)
        else:
            with multiprocessing.Pool(num_parallel_process) as p:
                _ = p.map(create_webdataset, input_args)
                
