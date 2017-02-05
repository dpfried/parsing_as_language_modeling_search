#!/bin/bash
# Job name:
#SBATCH --job-name=test
#SBATCH --account=fc_bnlp
#SBATCH --partition=savio2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mail-user=dfried@berkeley.edu
#SBATCH --mail-type=all
#

export MODULEPATH=$MODULEPATH:/global/home/groups/fc_bnlp/software/modfiles
module load tensorflow/unstable

candidate_file=$1

output_file=${candidate_file}.lstm-wsj-scores

python3 rescore.py \
  wsj/train_02-21.txt.traversed \
  models/wsj/model \
  $candidate_file \
  $output_file 
