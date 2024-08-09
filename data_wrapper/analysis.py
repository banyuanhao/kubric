import os

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from data_wrapper.utils import load_json

DATA_DIR = 'proc_data/data/ActivityNet'
RAW_VIDEO_DIR = os.path.join(DATA_DIR, 'videos')
PROC_VIDEO_DIR = os.path.join(DATA_DIR, 'videos_our_format')
os.makedirs(PROC_VIDEO_DIR, exist_ok=True)

# Load the ActivityNet captions
TRAIN_CAPS = load_json(os.path.join(DATA_DIR, 'train.json'))
VAL_CAPS = load_json(os.path.join(DATA_DIR, 'val_1.json'))
# TEST_CAPS = load_json(DATA_DIR, 'val_2.json')  # we won't use this
# as `val_1.json` is a superset of `val_2.json`


def show_stats(values, name, fontsize=12):
    # Show statistics of a list of values
    values = np.array(values)
    print(f'Mean: {values.mean()}')
    print(f'Median: {np.median(values)}')
    print(f'Std: {values.std()}')
    # Show quantiles
    for q in np.arange(0.1, 1, 0.1):
        print(f'{q * 100}th percentile: {np.percentile(values, q * 100)}')
    # Plot histogram, range is [0, 3*values.median()], count is log10 scaled
    plt.close('all')
    plt.hist(values, bins=50, range=(0, 3 * np.median(values)), log=True)
    plt.xlabel(name, fontsize=fontsize)
    plt.ylabel('Count', fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    # plt.show()
    plt.savefig(f'proc_data/data/ActivityNet/{name}-stats.png')


if __name__ == "__main__":
    # Count video length, event duration, number of events, etc.
    vid_lens, ev_durs, num_evs, ev_diffs = [], [], [], []
    for caps in tqdm(TRAIN_CAPS.values(), total=len(TRAIN_CAPS)):
        start_ts = np.array([ts[0] for ts in caps["timestamps"]])
        end_ts = np.array([ts[1] for ts in caps["timestamps"]])
        assert len(caps["sentences"]) == len(start_ts) == len(end_ts)
        vid_lens.append(caps["duration"])
        ev_durs.extend((end_ts - start_ts).tolist())
        num_evs.append(len(start_ts))
        # Check if there are overlapping events
        ev_diffs.extend((start_ts[1:] - end_ts[:-1]).tolist())

    print(f'Number of videos: {len(TRAIN_CAPS)}')
    print('Video length (s):')
    show_stats(vid_lens, 'Video length (s)')
    print('Event duration (s):')
    show_stats(ev_durs, 'Event duration (s)')
    print('Number of events per video:')
    show_stats(num_evs, 'Number of events per video')
    # Check if there are overlapping events
    # print('Time difference between events (s):')
    # print(f'{np.array(ev_diffs).min()=}')
    breakpoint()
