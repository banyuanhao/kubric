#!/bin/bash

# Serial Number
SENUM=18

# Calculate start and end values
START=$((SENUM * 1000))
END=$(((SENUM + 1) * 1000 - 1))

# Loop from START to END
for ((ii=START; ii<=END; ii++))
do
  echo "Running with ii=$ii"
  /usr/bin/python3 generated_dataset/simple_64f/worker.py --ii $ii --output_dir /nfs/data/banyuanhao/video/simple_64f/
done