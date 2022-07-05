#!/bin/bash

#SBATCH -J GetImages.py
#SBATCH -o GetImages.txt
#SBATCH -p gpu2v100
#SBATCH -t 40:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=xi.tang@my.utsa.edu

module load anaconda3
conda activate get_facial_images
python GetImages.py StevenOHara
conda deactivate
