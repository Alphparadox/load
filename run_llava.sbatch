#!/bin/bash

# -----------------
# SBATCH DIRECTIVES
# -----------------
# Job name
#SBATCH --job-name=llava_text_test
#
# Output and error log files
#SBATCH --output=llava_test_%j.out
#SBATCH --error=llava_test_%j.err
#
# Resource requests
#SBATCH --nodes=1                 # Request one node
#SBATCH --ntasks-per-node=1       # Request one task (process)
#SBATCH --cpus-per-task=4         # Request 4 CPUs per task (for data handling)
#SBATCH --mem=32G                 # Request 32GB of RAM (for the large model)
#
# --- THIS LINE IS NOW FIXED ---
#SBATCH --gres=gpu:shard:1        # Request 1 GPU using the 'shard' format
#
#SBATCH --time=00:15:00           # Request 15 minutes (should be enough)

# -----------------
# JOB ENVIRONMENT
# -----------------
echo "Starting job $SLURM_JOB_ID on $(hostname)"
echo "Loading environment..."

# --- IMPORTANT: ADJUST THESE LINES FOR YOUR CLUSTER ---

# 1. Load modules (if your cluster uses them)
# These names are examples; check your cluster's documentation
module purge
module load python/3.10
module load cuda/11.8

# 2. Activate your Python virtual environment
# --- !! YOU MUST CHANGE THIS PATH !! ---
source /path/to/your/virtualenv/bin/activate

echo "Environment loaded."

# -----------------
# RUN THE SCRIPT
# -----------------
echo "Running Python script..."

# --- !! YOU MUST CHANGE THIS NAME !! ---
# Make sure 'your_script_name.py' is the name of your Python file
python your_script_name.py

echo "Job finished."