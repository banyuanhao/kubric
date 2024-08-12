from datetime import datetime
from pytz import timezone
import multiprocessing
import os
import boto3
import argparse
import random
from tqdm import tqdm

from utils import compress_into_tar, load_json

DATA_DIR = '/fsx/yban/myproject/kubric/generated_dataset/reverse_time'
VIDEO_DIR = os.path.join(DATA_DIR, 'videos_our_format')
SHARD_DIR = os.path.join(DATA_DIR, 'shards_our_format')
os.makedirs(SHARD_DIR, exist_ok=True)
TRAIN_CAPS = load_json(os.path.join(DATA_DIR, 'train.json'))
VAL_CAPS = load_json(os.path.join(DATA_DIR, 'val.json'))


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
        assert all([os.path.isfile(i) for i in upload_list_per_video])
        upload_list += [i for i in upload_list_per_video if os.path.isfile(i)]

        # Gets the size of the current video
        video_size = os.path.getsize(os.path.join(VIDEO_DIR, video_filename))
        video_size_mb = video_size / (1024**2)
        shard_size += video_size_mb

        if shard_size > shard_size_mb:
            # Create shard
            compress_into_tar(upload_list, os.path.join(SHARD_DIR, shard_filename))

            # Uploads the shard to S3
            s3.upload_file(os.path.join(SHARD_DIR, shard_filename), "snap-webdataset-videos", "synthetic_one_direction" + shard_filename)
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
        s3.upload_file(os.path.join(SHARD_DIR, shard_filename), "snap-webdataset-videos", "synthetic_one_direction/" + shard_filename)
        with open(output_log_file, "a") as f:
            video_count = sum(1 for video_file in upload_list if video_file.endswith(".mp4"))
            f.write(datetime.now(timezone("US/Pacific")).strftime("%m/%d %H:%M:%S") + " %s (including %i videos) is uploaded\n"%(shard_filename, video_count))


if __name__ == "__main__":
    # python upload_activitynet_webdataset.py --split train
    # python upload_activitynet_webdataset.py --split val
    parser = argparse.ArgumentParser(description="Create webdataset")
    parser.add_argument("--split", type=str, required=True, choices=["train", "val"])
    parser.add_argument("--node-id", type=int, default=0)
    parser.add_argument("--num-node", type=int, default=1)
    parser.add_argument("--shard-size-mb", type=int, default=1000)
    args = parser.parse_args()

    # read valid clips
    train_keys = sorted(list(TRAIN_CAPS.keys()))
    val_keys = sorted(list(VAL_CAPS.keys()))
    print("Number of videos:")
    print(f"\tTrain: {len(train_keys)}\n\tVal: {len(val_keys)}")

    if args.split == "train":
        video_list = train_keys
        video_list = ["synthetic_one_direction_%s.mp4" % (vn[2:]) for vn in video_list]
        node_id = args.node_id
    elif args.split == "val":
        video_list = val_keys
        video_list = ["synthetic_one_direction_%s.mp4" % (vn[2:]) for vn in video_list]
        node_id = args.node_id + 100
    else:
        raise NotImplementedError
    output_log_file = os.path.join(SHARD_DIR, f"{args.split}_logs.txt")

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

    with multiprocessing.Pool(num_parallel_process) as p:
        _ = p.map(create_webdataset, input_args)
