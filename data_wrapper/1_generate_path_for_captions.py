import csv

# 文件名
csv_file = '/fsx/yban/myproject/kubric/generated_dataset/reverse_time/output.csv'

# 要生成的文件数量
num_files = 2000  # 你可以修改这个数字

# 打开文件并写入数据
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    
    # 写入标题行
    writer.writerow(['path'])
    
    # 写入文件名行
    for n in range(num_files):
        filename = f'/fsx/yban/myproject/kubric/generated_dataset/reverse_time/videos/video_{str(n).zfill(5)}.mp4'
        # filename = f'/fsx/yban/myproject/Open-Sora/samples/samples/sample_{str(n).zfill(4)}.mp4'
        writer.writerow([filename])

print(f"CSV文件 '{csv_file}' 已生成。")

# 2
# torchrun --nproc_per_node 1 --standalone -m tools.caption.caption_llava \
#   /fsx/yban/myproject/kubric/generated_dataset/reverse_time/output.csv \
#   --dp-size 1 \
#   --tp-size 1 \
#   --model-path liuhaotian/llava-v1.6-mistral-7b \
#   --prompt video
