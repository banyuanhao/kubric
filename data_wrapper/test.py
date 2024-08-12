from utils import load_pickle
file = load_pickle('/nfs/yban/myproject/kubric/generated_dataset/reverse_time/videos_our_format/synthetic_one_direction_video_00000.summary_text_embeddings.pkl')
print(file.keys())
tensor = file['embeddings']
print(tensor.device)