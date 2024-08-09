import pickle

# Define the file path
file_path = '/fsx/yban/myproject/kubric/generated_dataset/reverse_time/videos_our_format/synthetic_one_direction_video_00000.summary_text_embeddings.pkl'

# Open and load the pickle file
with open(file_path, 'rb') as file:
    data = pickle.load(file)

# Now 'data' contains the contents of the pickle file
print(data['eot_location_lst'])