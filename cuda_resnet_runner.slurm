#!/bin/bash

#SBATCH -J cuda_resnet_runner.py
#SBATCH -o cuda_resnet_runner.txt
#SBATCH -p gpu2v100
#SBATCH -t 20:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=xi.tang@my.utsa.edu

module load anaconda3
yes | conda create -n cuda_aibias python=3.8.8
conda activate cuda_aibias

pip install -r required_libraries.txt
python cuda_resnet_runner.py

conda deactivate
yes | conda env remove -n cuda_aibias
