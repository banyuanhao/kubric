import os
import glob
import argparse
import torch
from tqdm import tqdm
import sys
sys.path.append('/fsx/yban/intern/large-scale-video-generation')
from models.language.backbones.t5_language_model import T5LanguageModel
from utils import load_json, dump_pickle
import pandas as pd

DATA_DIR = '/fsx/yban/myproject/kubric/generated_dataset/reverse_time/videos'
DATA_CAPTION_DIR = '/fsx/yban/myproject/kubric/generated_dataset/reverse_time/output_caption_part0.csv'

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process videos")
    parser.add_argument("--node-id", type=int, default=0)
    parser.add_argument("--num-node", type=int, default=1)
    args = parser.parse_args()
    
    model_config = {
        "frozen": True,
        "variant": "t5-11b",
        "max_length": 128
    }
    language_model = T5LanguageModel(model_config)
    language_model.to("cuda")
    batch_size = 512
    
    df = pd.read_csv(DATA_CAPTION_DIR)

    summary_text = df.text.tolist()
    print(summary_text[0])
    exit()
    path = df.path
    
    all_embeds, all_eot_locs, all_token_ids = [], [], []
    summary_text_batches = [summary_text[i:i+batch_size] for i in range(0, len(summary_text), batch_size)]
    for summary_text_batch in tqdm(summary_text_batches):
        with torch.no_grad():
            embedding_results = language_model(summary_text_batch, torch.float32)
        embeddings = embedding_results["text_embeddings"].cpu().numpy()
        embeddings = [emb for emb in embeddings]  # make a list of np.ndarray
        eot_locations = embedding_results["eot_locations"]
        token_ids = embedding_results["token_ids"]

        all_embeds.extend(embeddings)
        all_eot_locs.extend(eot_locations)
        all_token_ids.extend(token_ids)

    for i, filename in enumerate(path):
        result = {
            "embeddings": all_embeds[i],  # list of np.ndarray of shape [L (128), C (1024)]
            "eot_location": all_eot_locs[i],  # list of int
            "token_ids": all_token_ids[i],  # list of list of int
            "model_name": "t5-11b"
        }
        output_filename = filename.replace(".mp4", "_embeddings.pkl")
        dump_pickle(result, output_filename)
