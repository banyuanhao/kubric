for i in $(seq $1 $2); do
    python generated_dataset/simple_64f/worker.py --output_dir /fsx_ori/yban/dataset/simple_64f/ --ii $i
done