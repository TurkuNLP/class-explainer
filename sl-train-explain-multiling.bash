#!/bin/bash
#SBATCH --job-name=training
#SBATCH --account=project_2002026
#SBATCH --time=20:15:00
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err

rm logs/current.err
rm logs/current.out
ln -s $SLURM_JOBID.err logs/current.err
ln -s $SLURM_JOBID.out logs/current.out

module purge
#module load pytorch/1.8
#source /scratch/project_2002026/amanda/venv/bin/activate
source /scratch/project_2002026/samuel/VENVS/expl/bin/activate

srun python run_resplits_multiling.py \
  --data oscar_data \
  --model_name xlm-roberta-base \
  --lr 7.5e-5 \
  --epochs 12 \
  --batch_size 30 \
  --split 0.9999 \
  --patience 1 \
  --save_explanations explanations/$1 \
  --save_model models/$1 \
  --seed $2

rm -vrf models/$1-ckpt
gzip explanations/$1.tsv 

seff $SLURM_JOBID
