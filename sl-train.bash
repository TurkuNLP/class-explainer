#!/bin/bash
#SBATCH --job-name=training
#SBATCH --account=project_2002026
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=7G
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err

mkdir -p logs
rm logs/current.err
rm logs/current.out
ln -s $SLURM_JOBID.err logs/current.err
ln -s $SLURM_JOBID.out logs/current.out

module purge
module load pytorch/1.8
source /scratch/project_2002026/amanda/venv/bin/activate
srun python train_multilabel.py \
  --data binarized_data/eacl_en_binarized.pkl \
  --model_name xlm-roberta-base \
  --lr 1e-5 \
  --epochs 6 \
  --batch_size 28 \
  --ratio 0.5 \
  --checkpoints multilabel_model_checkpoints \
  --save_model models/multilabel_model3_fifrsv.pt

seff $SLURM_JOBID
