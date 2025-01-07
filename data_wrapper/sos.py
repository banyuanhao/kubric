import os
from concurrent.futures import ThreadPoolExecutor

# 文件夹路径
folder_path = '/nfs/yban/kubric/generated_dataset/reverse_time_complex/videos/'

# 获取文件夹下所有文件
files = os.listdir(folder_path)

def rename_file(file_name):
    # 检查文件是否以 metadata 开头并且是 .pkl 文件
    if file_name.startswith('metadata') and file_name.endswith('.pkl'):
        # 生成新的文件名，将 metadata 替换为 video
        new_file_name = file_name.replace('metadata', 'video', 1)
        # 获取完整的旧文件路径和新文件路径
        old_file_path = os.path.join(folder_path, file_name)
        new_file_path = os.path.join(folder_path, new_file_name)
        # 重命名文件
        os.rename(old_file_path, new_file_path)
        print(f'Renamed: {old_file_path} -> {new_file_path}')

# 使用 ThreadPoolExecutor 进行多线程重命名操作
with ThreadPoolExecutor(max_workers=32) as executor:
    # 提交任务给线程池
    executor.map(rename_file, files)
