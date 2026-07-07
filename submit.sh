#!/bin/bash --login
#SBATCH -p gpuL
#SBATCH -G 1
#SBATCH -t 1-0
#SBATCH -c 12

cp /mnt/iusers01/fse-ugpgt01/phy01/m64196am/alexei_projects/data/ttbar_test.h5 ~/scratch 
cp /mnt/iusers01/fse-ugpgt01/phy01/m64196am/alexei_projects/data/ttbar_train.h5 ~/scratch 
cp /mnt/iusers01/fse-ugpgt01/phy01/m64196am/alexei_projects/data/ttbar_val.h5 ~/scratch 
cp /mnt/iusers01/fse-ugpgt01/phy01/m64196am/alexei_projects/data/scaler_info.h5 ~/scratch 
cp /mnt/iusers01/fse-ugpgt01/phy01/m64196am/alexei_projects/scripts/ttbar_train_transformer.py ~/scratch
cd ~/scratch
source ~/.venv/bin/activate
python ttbar_train_transformer.py
