import glob
import pandas as pd
import os
import argparse
from tqdm import tqdm
from multiprocessing import Pool


from utils import load_json, dump_json, get_video_information, reencode_video, get_image_information, reencode_image

DATA_CAPTION_DIR = 'generated_dataset/multiple_64f/metadata_splited.csv'



DATA_DIR = '/fsx_ori/yban/dataset/multiple_64f'
PROC_IMAGE_DIR = os.path.join(DATA_DIR, 'images_our_format')
os.makedirs(PROC_IMAGE_DIR, exist_ok=True)



def process_video(video_path):
    """Process on video file.

    We will reencode it for faster decoding, and save its metadata (e.g.,
        framerate, frames count, duration, width, height) to a json file.
    In addition, we will process its text captions to a json file.
    """
    vidinfo = {}
    video_name = 'image_' + os.path.basename(os.path.dirname(video_path)) + '_' + os.path.basename(video_path).split('_')[-1].replace(".png", "")
    
    dir_path = os.path.dirname(os.path.dirname(video_path)).replace("rawdata", "videos")
    id_index = os.path.basename(os.path.dirname(video_path))
    embeddings_file = f"video_{id_index}_embeddings_image.pkl"
    embeddings_path = os.path.join(dir_path, embeddings_file)
    
    
    vidinfo["dataset"] = "multi_image"
    vidinfo["pre_encode_video_filename"] = video_name + ".png"
    vidinfo["filename"] = "multi_%s.png" % (video_name)

    # Check if the video has already been processed
    output_name = os.path.join(PROC_IMAGE_DIR, vidinfo["filename"])
    info_file = output_name.replace(".png", ".imginfo.json")
    caps_file = output_name.replace(".png", ".summary_text.json")
    embeddings_file = output_name.replace(".png", ".summary_text_embeddings.pkl")
    
    if not os.path.exists(embeddings_path):
        print(f"Embeddings file {embeddings_path} not found. Skipping...")
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

    # get video framerate / frames count / duration / width / height
    results = get_image_information(video_path)

    vidinfo["original_width"] = results["width"]
    vidinfo["original_height"] = results["height"]


    # reencode video
    # add exception handling for reencode_image
    try:
        vidinfo["width"], vidinfo["height"] = reencode_image(video_path, output_name)
    except:
        print(f"Error in reencoding image {video_path}. Skipping...")
        os.system("rm -rf %s" % output_name)
        os.system("rm -rf %s" % info_file)
        os.system("rm -rf %s" % caps_file)
        return

    summary_text = {
        "text": dct[video_path],
    }

    # output vidinfo and summary_text
    dump_json(vidinfo, info_file)
    dump_json(summary_text, caps_file)
    
    os.system("cp %s %s" % (embeddings_path, embeddings_file))


def process_video_group(video_paths):
    for video_path in tqdm(video_paths):
        process_video(video_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process videos")
    parser.add_argument("--node-id", type=int, default=0)
    parser.add_argument("--num-node", type=int, default=1)
    args = parser.parse_args()
    
    data = pd.read_csv(DATA_CAPTION_DIR)
    print(data.head())
    select_data_image = data[(data['usage'] == 'train_image') |  (data['usage'] == 'val_image')]
    print(f"Number of videos: {len(select_data_image)}")
    

    # We will go over all videos
    image_paths = select_data_image.image_path.tolist()
    print(len(image_paths))
    image_paths = [eval(image_path) for image_path in image_paths]
    image_paths = [item for sublist in image_paths for item in sublist]
    print(len(image_paths))
    caps = select_data_image.text_video.tolist()
    # repeat the captions 8 times to match the number of frames
    caps = [cap for cap in caps for _ in range(8)]
    print(len(image_paths))
    print(len(caps))
    dct = dict(zip(image_paths, caps))
    
    image_paths.sort()
    image_paths = image_paths[args.node_id::args.num_node]
    print(f"Processing {len(image_paths)} videos...")

    # multi-process each movie
    num_processes = 32
    grouped_image_paths = [image_paths[i::num_processes] for i in range(num_processes)]
    with Pool(num_processes) as p:
        _ = p.map(process_video_group, grouped_image_paths)
