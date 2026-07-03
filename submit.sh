#!/bin/bash --login
#SBATCH -p gpuL
#SBATCH -G 1
#SBATCH -t 1-0
#SBATCH -c 12

source .venv/bin/activate
python ML-TTBar_Regression.py
