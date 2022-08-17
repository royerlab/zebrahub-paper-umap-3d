#! /bin/bash

#SBATCH --job-name=rot3dumap
#SBATCH --time=24:00:00
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=128GB


env | grep "^SLURM" | sort

module load anaconda
conda activate napari3drot

python rot3dumap_by_embryo.py