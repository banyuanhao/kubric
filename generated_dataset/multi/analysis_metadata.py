import pandas as pd
import json
import os

from moviepy.editor import VideoFileClip, clips_array, vfx

def create_video_grid(video_paths, output_path, h=64, w=64):
    """
    Create a grid of videos and save the result as an mp4 video.

    Parameters:
    video_paths (list of str): List of paths to the input video files.
    output_path (str): Path to save the output video.
    h (int): Height of each individual video.
    w (int): Width of each individual video.

    Returns:
    None
    """

    # Check if the number of videos is a multiple of 4
    if len(video_paths) % 4 != 0:
        raise ValueError("Number of video files must be a multiple of 4.")

    n = len(video_paths) // 4  # Number of columns in the grid

    # Load videos and resize them
    video_clips = [VideoFileClip(video_path).resize((w, h)) for video_path in video_paths]

    # Create the grid array
    grid = []
    for i in range(0, len(video_clips), n):
        grid.append(video_clips[i:i+n])  # Add each row of videos to the grid

    # Combine videos into a grid
    final_clip = clips_array(grid)

    # Save the resulting video to the output path
    final_clip.write_videofile(output_path, codec="libx264")
    
split_sets = json.load(open('generated_dataset/multi/split_sets.json'))

data = pd.read_csv('generated_dataset/multi/metadata_splited.csv')
rows = data[(data['direction'] == 'left') & (data['usage'] == 'train_video')]
data_path = '/nfs/yban/dataset/multi/videos_our_format'

videos_path = [os.path.join(data_path, f'multi_video_{id:05d}.mp4') for id in rows.id.to_list()]

videos_info_path = [os.path.join(data_path, f'multi_video_{id:05d}.summary_text.json') for id in rows.id.to_list()]

# for i in range(0, len(videos_path), 16):
#     output_path = f'generated_dataset/multi/videos/videos_grid_{i}.mp4'
#     create_video_grid(videos_path[i:i+16], output_path)
#     print(f"Created video grid: {output_path}")

for video_info_path in videos_info_path:
    with open(video_info_path) as f:
        video_info = json.load(f)['text']
    print(video_info)
    
    if 'guide' in video_info:
        print(video_info)
        raise ValueError('Left in video info')