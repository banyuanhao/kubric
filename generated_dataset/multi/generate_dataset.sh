for i in $(seq $1 $2); do
    python generated_dataset/multi/worker.py --output_dir /nfs/yban/dataset/multi --ii $i
done

# /nfs/yban/kubric/setup.sh

# python generated_dataset/reverse_time_video_and_image_20/worker.py --output_dir generated_dataset/reverse_time_video_and_image_20/test --ii 1 --split video_and_image --direction right
# ./generated_dataset/reverse_time_video_and_image_20/generate_dataset.sh 110 120 video_and_image left

# ./generated_dataset/multi/generate_dataset.sh 0 250
# ./generated_dataset/multi/generate_dataset.sh 250 500
# ./generated_dataset/multi/generate_dataset.sh 500 750
# ./generated_dataset/multi/generate_dataset.sh 750 1000
# ./generated_dataset/multi/generate_dataset.sh 1000 1250
# ./generated_dataset/multi/generate_dataset.sh 1250 1500
# ./generated_dataset/multi/generate_dataset.sh 1500 1750
# ./generated_dataset/multi/generate_dataset.sh 1750 2000
# ./generated_dataset/multi/generate_dataset.sh 2000 2250
# ./generated_dataset/multi/generate_dataset.sh 2250 2500
# ./generated_dataset/multi/generate_dataset.sh 2500 2750
# ./generated_dataset/multi/generate_dataset.sh 2750 3000
# ./generated_dataset/multi/generate_dataset.sh 3000 3250
# ./generated_dataset/multi/generate_dataset.sh 3250 3500
# ./generated_dataset/multi/generate_dataset.sh 3500 3750
# ./generated_dataset/multi/generate_dataset.sh 3750 4000
# ./generated_dataset/multi/generate_dataset.sh 4000 4250
# ./generated_dataset/multi/generate_dataset.sh 4250 4500
# ./generated_dataset/multi/generate_dataset.sh 4500 4750
# ./generated_dataset/multi/generate_dataset.sh 4750 5000
# ./generated_dataset/multi/generate_dataset.sh 5000 5250
# ./generated_dataset/multi/generate_dataset.sh 5250 5500
# ./generated_dataset/multi/generate_dataset.sh 5500 5750
# ./generated_dataset/multi/generate_dataset.sh 5750 6000
# ./generated_dataset/multi/generate_dataset.sh 6000 6250
# ./generated_dataset/multi/generate_dataset.sh 6250 6500
# ./generated_dataset/multi/generate_dataset.sh 6500 6750
# ./generated_dataset/multi/generate_dataset.sh 6750 7000
# ./generated_dataset/multi/generate_dataset.sh 7000 7250
# ./generated_dataset/multi/generate_dataset.sh 7250 7500
# ./generated_dataset/multi/generate_dataset.sh 7500 7750
# ./generated_dataset/multi/generate_dataset.sh 7750 8000