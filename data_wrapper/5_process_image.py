import glob
import os
import argparse
from tqdm import tqdm
from multiprocessing import Pool

from utils import load_json, dump_json, reencode_image, get_resolution_image, get_image_information

DATA_DIR = '/fsx/yban/kubric/generated_dataset/reverse_time_video_and_image'
RAW_VIDEO_DIR = os.path.join(DATA_DIR, 'images')
PROC_VIDEO_DIR = os.path.join(DATA_DIR, 'images_our_format_two_direction')
os.makedirs(PROC_VIDEO_DIR, exist_ok=True)

# Load the ActivityNet captions
TRAIN_CAPS = load_json(os.path.join(DATA_DIR, 'images_train_two_direction.json'))
VAL_CAPS = load_json(os.path.join(DATA_DIR, 'images_val_two_direction.json'))
# TEST_CAPS = load_json(DATA_DIR, 'val_2.json')  # we won't use this
# as `val_1.json` is a superset of `val_2.json`


def process_video(video_path):
    """Process on video file.

    We will reencode it for faster decoding, and save its metadata (e.g.,
        framerate, frames count, duration, width, height) to a json file.
    In addition, we will process its text captions to a json file.
    """
    vidinfo = {}
    video_name = os.path.basename(video_path).replace(".png", "")
    vidinfo["dataset"] = "synthetic_one_direction_video_and_image"
    vidinfo["pre_encode_video_filename"] = video_name + ".png"
    vidinfo["filename"] = "synthetic_one_direction_video_and_image_%s.png" % (video_name)

    # Check if the video has already been processed
    output_name = os.path.join(PROC_VIDEO_DIR, vidinfo["filename"])
    info_file = output_name.replace(".png", ".imginfo.json")
    caps_file = output_name.replace(".png", ".summary_text.json")
    embeddings_file = output_name.replace(".png", ".summary_text_embeddings.pkl")
    
    if not os.path.exists(video_path.replace(".png", "_embeddings.pkl")):
        p = video_path.replace(".png", "_embeddings.pkl")
        print(f"Embeddings file {p} not found. Skipping...")
        return
    
    if os.path.exists(info_file) and os.path.exists(caps_file):
        # Check if they are not corrupted
        info = load_json(info_file)
        caps = load_json(caps_file)
        if 'width' in info and 'text_lst' in caps:
            print(f"Video {video_name} already processed. Skipping...")
            return
    os.system("rm -rf %s" % output_name)
    os.system("rm -rf %s" % info_file)
    os.system("rm -rf %s" % caps_file)

    # if video name is "QOlSCBRmfWY.mp4", then caption key is "v_QOlSCBRmfWY"
    cap_key = "v_" + video_name
    if cap_key in TRAIN_CAPS.keys():
        caps = TRAIN_CAPS[cap_key]
    elif cap_key in VAL_CAPS.keys():
        caps = VAL_CAPS[cap_key]
    else:
        print(f"Video {video_name} not found in captions. Skipping...")
        return

    # get video framerate / frames count / duration / width / height
    results = get_image_information(video_path)

    vidinfo["original_width"] = results["width"]
    vidinfo["original_height"] = results["height"]


    # reencode video
    vidinfo["width"], vidinfo["height"] = reencode_image(video_path, output_name)

    summary_text = {
        "text": caps
    }

    # output vidinfo and summary_text
    dump_json(vidinfo, info_file)
    dump_json(summary_text, caps_file)
    
    os.system("cp %s %s" % (video_path.replace(".png", "_embeddings.pkl"), embeddings_file))


def process_video_group(video_paths):
    for video_path in tqdm(video_paths):
        process_video(video_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process videos")
    parser.add_argument("--node-id", type=int, default=0)
    parser.add_argument("--num-node", type=int, default=1)
    args = parser.parse_args()

    # We will go over all videos
    video_paths = glob.glob(os.path.join(RAW_VIDEO_DIR, '*.png'))
    video_paths.sort()
    video_paths = video_paths[args.node_id::args.num_node]
    print(f"Processing {len(video_paths)} images...")

    # multi-process each movie
    num_processes = 32
    grouped_video_paths = [video_paths[i::num_processes] for i in range(num_processes)]
    with Pool(num_processes) as p:
        _ = p.map(process_video_group, grouped_video_paths)
