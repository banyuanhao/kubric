for i in $(seq $1 $2); do
    python generated_dataset/multiple/worker.py --output_dir /nfs/yban/dataset/multiple --ii $i
done