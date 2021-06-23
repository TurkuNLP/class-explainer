#!/bin/bash
#SBATCH --job-name=explain
#SBATCH --account=project_2002026
#SBATCH --time=00:15:00
#SBATCH --partition=gputest
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=7G
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
module load pytorch/1.8
source /scratch/project_2002026/amanda/venv/bin/activate
srun python3 explain_multilabel.py \
  --model_name models/xxxx.pt     \
  --data ../../veronika/simplified-data  \
  --int_batch_size 20 \
  --file_name keywords/xxxx


seff $SLURM_JOBID
