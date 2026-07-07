#!/bin/bash --login
#SBATCH -p gpuL
#SBATCH -G 2
#SBATCH -t 1-0
#SBATCH -c 12

cp /mnt/iusers01/fse-ugpgt01/phy01/q44882cb/HEP_Sigmoids/train_inputs/{ttbar_train.h5,ttbar_test.h5,ttbar_val.h5,scaler_info.h5,feature_labels.h5} ~/scratch/train_inputs/

cp /mnt/iusers01/fse-ugpgt01/phy01/q44882cb/HEP_Sigmoids/Scripts/ttbar_train_transformer.py ~/scratch/Scripts/

cd ~/scratch/Scripts
source ~/.venv/bin/activate

srun python ttbar_train_transformer.py