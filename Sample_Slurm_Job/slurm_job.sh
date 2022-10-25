#!/bin/bash
#
# Sample HPC SLURM Job Script
# by Arda Mavi
#
#SBATCH --account=<account>
#SBATCH --job-name=<job_name>
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=4gb
#SBATCH --time=<hours>:00:00
#SBATCH --partition=<partition>
#SBATCH -o output.out
#SBATCH --open-mode=append
#SBATCH --mail-type=END
#SBATCH --mail-user=<email_address>
#SBATCH --export=all
#SBATCH --chdir=../Main\ Pipeline

conda activate <envoriment_name>
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<cuda_path>

echo "Job Started"

python main_pipeline.py

echo "Job Ended"
