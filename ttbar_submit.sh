#!/bin/bash --login
#SBATCH -p gpuL
#SBATCH -G 2
#SBATCH -t 1-0
#SBATCH -c 12

echo "# ---------- BATCH RECIEPT ---------- #"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Submit Time: $(date)"
echo "Node: $(hostname)"
echo "User: $USER"
echo "Partition: $SLURM_JOB_PARTITION"
echo "GPUs: $SLURM_GPUS"
echo "Working Directory: $(pwd)"
echo "# ----------------------------------- #"
echo ""


cp /mnt/iusers01/fse-ugpgt01/phy01/q44882cb/HEP_Sigmoids/train_inputs/{larger_ttbar_train.h5,larger_ttbar_test.h5,larger_ttbar_val.h5,larger_scaler_info.h5,feature_labels.h5} ~/scratch/train_inputs/

cp /mnt/iusers01/fse-ugpgt01/phy01/q44882cb/HEP_Sigmoids/Scripts/ttbar_train_transformer.py ~/scratch/Scripts/

cd ~/scratch/Scripts
source ~/.venv/bin/activate

srun python ttbar_train_transformer.py