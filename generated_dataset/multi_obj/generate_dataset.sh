for i in $(seq $1 $2); do
    python generated_dataset/multi_obj/worker.py --output_dir /nfs/yban/dataset/multi_obj --ii $i --camera=linear_movement_linear_lookat --max_camera_movement=8.0
done