#!/bin/bash

# -----------------
# SBATCH DIRECTIVES
# -----------------
# Job name
#SBATCH --job-name=llava_text_test
#
# --- FIXED: Using your working log paths ---
#SBATCH --output=../logs/llava_test_%j.out
#SBATCH --error=../logs/llava_test_%j.err
#
# Resource requests (from your working file)
#SBATCH --ntasks=1                 # Use ntasks=1 (simpler than nodes/ntasks-per-node)
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#
# --- CRITICAL: Added from your working file ---
#SBATCH --partition=gpu
#SBATCH --gres=shard:1
#SBATCH --exclude=node1
#
#SBATCH --time=01:00:00            # 15 minutes is fine for this test

# -----------------
# JOB ENVIRONMENT
# -----------------
echo "Starting job $SLURM_JOB_ID on $(hostname)"

# --- FIXED: Using your working conda environment ---
echo "Activating conda environment..."
source ~/miniconda3/bin/activate kiva_env

echo "Environment loaded."
echo "Running on: $HOSTNAME"
nvidia-smi

# -----------------
# RUN THE SCRIPT
# -----------------
echo "Running Python script..."

# --- !! YOU MUST CHANGE THIS NAME !! ---
# Make sure 'your_script_name.py' is the name of your Python file
# (e.g., test_model.py, or whatever you called the "capital of India" script)

python sample.py --benchmark_file /fyp/harsh/benchmark_data.json --image_base_dir ~/fyp/

echo "Job finished."