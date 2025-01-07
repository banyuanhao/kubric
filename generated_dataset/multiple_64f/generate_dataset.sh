for i in $(seq $1 $2); do
    python generated_dataset/multiple_64f/worker.py --output_dir /fsx_ori/yban/dataset/multiple_64f/ --ii $i
done


docker run --rm --interactive \
           --user $(id -u):$(id -g) \
           --volume "$(pwd):/kubric" \
           kubricdockerhub/kubruntu \
           /usr/bin/python3 generated_dataset/multiple_64f/worker.py --output_dir /fsx_ori/yban/dataset/multiple_64f/ 1