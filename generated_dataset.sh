for i in {1750..1999}; do
    python generated_dataset/reverse_time/movi_ab_worker.py --output_dir generated_dataset/reverse_time --ii $i
done
