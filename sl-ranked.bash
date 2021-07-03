#!/bin/bash
#SBATCH --job-name=kw_rank
#SBATCH --account=project_2002026
#SBATCH --time=12:10:00
#SBATCH --partition=small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=5G
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
srun python ranked_keywords.py \
  --data ../../samuel/class-explainer/explanations/quickrunF0 \
  --language "" \
  --choose_best 20 \
  --drop_amb 3 \
  --fraction 0.8 \
  --save_n 50 \
  --save_file kw_results/kw_ranked.tsv

seff $SLURM_JOBID
