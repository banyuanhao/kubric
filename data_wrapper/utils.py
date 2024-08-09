import os
import gzip
import json
import pickle
import tarfile
import subprocess
import boto3
import botocore
from moviepy.editor import VideoClip, VideoFileClip, ImageSequenceClip, TextClip, CompositeVideoClip, concatenate_videoclips


########################################
# S3-related functions
########################################


def s3_file_exists(bucket, folder, filename, log_file=None, s3=None):
    if s3 is None:
        s3 = boto3.client("s3")
    key = os.path.join(folder, filename)
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except botocore.exceptions.ClientError as e:
        if e.response["Error"]["Code"] == "404" and log_file is not None:
            with open(log_file, "a") as f:
                f.write(key + " does not exist in s3 bucket\n")
        return False


def s3_download_file(bucket, folder, filename, local_folder, log_file=None, s3=None):
    if s3 is None:
        s3 = boto3.client("s3")
    key = os.path.join(folder, filename)
    local_path = os.path.join(local_folder, filename)
    if os.path.exists(local_path):
        return local_path
    try:
        s3.download_file(bucket, key, local_path)
        return local_path
    except Exception as e:
        # if the required file cannot be downloaded, raise error
        if log_file is not None:
            with open(log_file, "a") as f:
                f.write(key + " failed to download from s3 bucket\n")
                f.write(str(e) + "\n")
        else:
            print(key + " failed to download from s3 bucket")
            print(str(e))
        return False


def s3_upload_file(bucket, folder, local_file, s3=None):
    if s3 is None:
        s3 = boto3.client("s3")
    s3_path = os.path.join(folder, os.path.basename(local_file))
    s3.upload_file(local_file, bucket, s3_path)


def list_s3_files(bucket, folder):
    s3 = boto3.client('s3')
    paginator = s3.get_paginator('list_objects_v2')
    page_iterator = paginator.paginate(Bucket=bucket, Prefix=folder)
    files = []
    for page in page_iterator:
        if 'Contents' in page:
            for obj in page['Contents']:
                files.append(obj['Key'])
    return files


########################################
# IO-related functions
########################################


def read_all_lines(fn):
    with open(fn, 'r') as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    return lines


def write_to_lines(fn, lines):
    assert not os.path.exists(fn), f"{fn=} already exists"
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    with open(fn, 'w') as f:
        for line in lines:
            f.write(line + '\n')


def load_json(file):
    if isinstance(file, str):
        with open(file, 'r') as f:
            obj = json.load(f)
    elif hasattr(file, 'read'):
        obj = json.load(file)
    else:
        raise TypeError('"file" must be a filename str or a file-object')
    return obj


def dump_json(obj, file=None, **kwargs):
    if file is None:
        return json.dumps(obj, **kwargs)
    elif isinstance(file, str):
        with open(file, 'w') as f:
            json.dump(obj, f, **kwargs)
    elif hasattr(file, 'write'):
        json.dump(obj, file, **kwargs)
    else:
        raise TypeError('"file" must be a filename str or a file-object')


def load_pickle(file, **kwargs):
    if isinstance(file, str):
        if file.endswith('.pkl'):
            with open(file, 'rb') as f:
                obj = pickle.load(f, **kwargs)
        elif file.endswith('.pkl.gz'):
            with gzip.open(file, 'rb') as f:
                obj = pickle.load(f, **kwargs)
        else:
            raise ValueError('Unknown file extension: %s' % file)
    elif hasattr(file, 'read'):
        obj = pickle.load(file, **kwargs)
    else:
        raise TypeError('"file" must be a filename str or a file-object')
    return obj


def dump_pickle(obj, file=None, **kwargs):
    kwargs.setdefault('protocol', 2)
    if file is None:
        return pickle.dumps(obj, **kwargs)
    elif isinstance(file, str):
        with open(file, 'wb') as f:
            pickle.dump(obj, f, **kwargs)
    elif hasattr(file, 'write'):
        pickle.dump(obj, file, **kwargs)
    else:
        raise TypeError('"file" must be a filename str or a file-object')


def compress_into_tar(file_list, tar_file):
    with tarfile.open(tar_file, "w") as tar:
        for file_path in file_list:
            tar.add(file_path, arcname=os.path.basename(file_path))


def uncompress_tar(tar_file, output_dir):
    assert not os.path.exists(output_dir), f"{output_dir=} already exists"
    with tarfile.open(tar_file, "r") as tar:
        tar.extractall(path=output_dir)


########################################
# FFMPEG-related functions
########################################


def get_length(video_path):
    ffprobe_command = "/usr/bin/ffprobe"
    result = subprocess.run([ffprobe_command, "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", video_path], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return float(result.stdout)


def get_num_frames(video_path):
    ffprobe_command = "/usr/bin/ffprobe"
    pipe = subprocess.Popen([ffprobe_command, "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", video_path], stdout=subprocess.PIPE).stdout
    output = pipe.read()
    if output.isdigit():
        return int(output)
    else:
        return int(output.decode("utf-8").split(",")[0])


def get_fps(video_path):
    ffprobe_command = "/usr/bin/ffprobe"
    pipe = subprocess.Popen([ffprobe_command, "-v", "error", "-select_streams", "v", "-of", "default=noprint_wrappers=1:nokey=1", "-show_entries", "stream=r_frame_rate", video_path], stdout=subprocess.PIPE).stdout
    output = pipe.read()
    packed_output = output.decode("utf-8").split("/")
    if len(packed_output) != 2:
        raise ValueError("Unexpected output from ffprobe: '{}'. Video at path '{}' appears to be corrupted.".format(packed_output, video_path))
    numerator, denominator = packed_output
    return float(numerator) / float(denominator)


def get_resolution(video_path):
    ffprobe_command = "/usr/bin/ffprobe"
    pipe = subprocess.Popen([ffprobe_command, '-v', 'error', '-select_streams', 'v', '-of', 'default=noprint_wrappers=1:nokey=1', '-show_entries', 'stream=width,height', video_path], stdout=subprocess.PIPE).stdout
    output = pipe.read()
    width, height = output.decode("utf-8").split("\n")[:2]
    return int(width), int(height)


def concatenate_videos(video_splits, output_fn):
    tmp_fn = video_splits[-1].replace('.mp4', '.txt')
    with open(tmp_fn, 'w') as f:
        for fn in video_splits:
            f.write(f"file '{os.path.basename(fn)}'\n")
    os.system(f"/usr/bin/ffmpeg -f concat -safe 0 -i {tmp_fn} -hide_banner -loglevel quiet -c copy {output_fn}")
    os.system(f"rm -rf {tmp_fn}")


def reencode_video(video_path, output_name):
    os.system("""/usr/bin/ffmpeg -i %s -vcodec libx265 -hide_banner -loglevel quiet -x265-params log-level=quiet -pix_fmt yuv420p -crf 23 -g 5 -tune fastdecode -vf "scale='min(1024,iw)':-2" -vsync vfr %s""" % (video_path, output_name))
    width, height = get_resolution(output_name)
    return int(width), int(height)


def get_video_information(video_path):
    result = {}

    # get frames count
    result["frames_count"] = get_num_frames(video_path)

    # get framerate
    result["framerate"] = get_fps(video_path)

    # get duration
    result["duration"] = result["frames_count"] / result["framerate"]

    # get dimention
    result["width"], result["height"] = get_resolution(video_path)

    return result


def add_subtitle_to_video(
    video_fn, text_lst, start_ts, end_ts,
    fontsize=24, color='black', bg_color='transparent', font='DejaVu-Serif',
):
    if isinstance(video_fn, str):
        video = VideoFileClip(video_fn)
    else:
        assert isinstance(video_fn, VideoClip)
        video = video_fn
    video = video.without_audio()
    clips = []
    for text, start_t, end_t in zip(text_lst, start_ts, end_ts):
        video_clip = video.subclip(start_t, end_t)
        text_clip = TextClip(
            text, fontsize=fontsize, color=color, bg_color=bg_color, font=font,
            method='caption', align='South', size=(video.w, None),
        )
        text_clip = text_clip.set_pos(('center', 0.85 * video.h)).set_duration(video_clip.duration)
        video_clip = CompositeVideoClip([video_clip, text_clip])
        video_clip = video_clip.without_audio()
        clips.append(video_clip)
    video = concatenate_videoclips(clips)
    video.write_videofile(video_fn.replace('.mp4', '_caption.mp4'))


def vid2gif(video_fn, skip_rate=1):
    video = VideoFileClip(video_fn)
    frames = [f for f in video.iter_frames()]
    frames = frames[::skip_rate]
    gif = ImageSequenceClip(frames, fps=video.fps / skip_rate)
    gif.write_gif(video_fn.replace('.mp4', f'-skip_{skip_rate}.gif'))
