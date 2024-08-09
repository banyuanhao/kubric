import os
import glob
import argparse
import torch
from tqdm import tqdm
from opensora.models.text_encoder.t5 import T5Encoder
from utils import load_json, dump_pickle
import pandas as pd

DATA_DIR = '/fsx/yban/myproject/kubric/generated_dataset/reverse_time/videos'
DATA_CAPTION_DIR = '/fsx/yban/myproject/kubric/generated_dataset/reverse_time/output_caption_part0.csv'

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process videos")
    parser.add_argument("--node-id", type=int, default=0)
    parser.add_argument("--num-node", type=int, default=1)
    args = parser.parse_args()
    text_encoder = dict(
        from_pretrained="DeepFloyd/t5-v1_1-xxl",
        model_max_length=300,
        shardformer=False,
        device="cuda",
    )
    
    # TODO shardformer
    language_model = T5Encoder(device=text_encoder["device"], from_pretrained=text_encoder["from_pretrained"], model_max_length=text_encoder["model_max_length"],shardformer=text_encoder["shardformer"])
    batch_size = 512
    
    df = pd.read_csv(DATA_CAPTION_DIR)

    summary_text = df.text.tolist()
    path = df.path
    
    all_embeds, all_eot_locs, all_token_ids = [], [], []
    summary_text_batches = [summary_text[i:i+batch_size] for i in range(0, len(summary_text), batch_size)]
    for summary_text_batch in tqdm(summary_text_batches):
        with torch.no_grad():
            embedding_results = language_model.encode(summary_text_batch,required_input_ids=True)
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
