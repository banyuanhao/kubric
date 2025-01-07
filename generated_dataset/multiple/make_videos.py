import os
import cv2
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm

def write_video(data, output_path, fps=12):
    # Ensure data is in the correct shape
    assert data.ndim == 4, "Data should be a 4D numpy array"
    
    # Get the dimensions of the frames
    num_frames, height, width, channels = data.shape

    # Convert RGBA to RGB (discard the alpha channel)
    data_rgb = data[:, :, :, :3]
    
    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' for mp4 output
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for i in range(num_frames):
        # Ensure frame is in the correct type
        frame = data_rgb[i].astype(np.uint8)
        out.write(frame)

    out.release()

def process_video(name):
    video_path = os.path.join(base_path, name)
    if not os.path.isdir(video_path):
        return
    images = os.listdir(video_path)
    images = [image for image in images if 'rgba' in image]
    images.sort()
    images = [os.path.join(video_path, image) for image in images]
    # read rgba images and convert to a numpy array
    images = [cv2.imread(image) for image in images]
    images = np.array(images)
    
    output_path = os.path.join(os.path.dirname(base_path), 'videos', f'{name}.mp4')
    write_video(images, output_path)

if __name__ == '__main__':
    # base_path = '/nfs/yban/dataset/multiple/rawdata'
    # names = os.listdir(base_path)
    # names.sort()

    # with Pool() as pool:
    #     for _ in tqdm(pool.imap_unordered(process_video, names), total=len(names)):
    #         pass
    
    base_path = '/nfs/yban/dataset/multiple/videos'
    names = os.listdir(base_path)
    names.sort()
    for name in tqdm(names):
        if not name.endswith('.mp4'):
            continue
        src_path = os.path.join(base_path, name)
        dst_path = os.path.join(base_path, 'video_'+name)
        os.rename(src_path, dst_path)